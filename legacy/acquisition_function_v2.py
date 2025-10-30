"""
Enhanced Variance Reduction with Coverage Acquisition Function V2.

针对有限次数实验，优化主效应和交互效应估计精度的改进采集函数。

改进要点:
1. 避免重复采样 - 已采样设计直接排除或大幅惩罚
2. 提升空间覆盖度 - 多样性指标权重增强
3. 分区均匀性 - 对低覆盖区域提升权重
4. 信息增益优先 - 量化对主效应和交互效应方差的降低
5. 动态权重调整 - 根据估计精度动态调整策略
6. 兼顾高分探索 - 适度exploration以避免局部最优
"""

import numpy as np
import configparser
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from collections import Counter

# BoTorch imports
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform

# 支持相对导入和直接导入
try:
    from ..gower_distance import compute_coverage_batch, compute_coverage
    from ..gp_variance import GPVarianceCalculator
except ImportError:
    from gower_distance import compute_coverage_batch, compute_coverage
    from gp_variance import GPVarianceCalculator


class EnhancedVarianceReductionAcqf(AcquisitionFunction):
    """
    Enhanced Variance Reduction Acquisition Function for Experimental Design.

    针对有限次数实验的改进采集函数，目标是最大化主效应和交互效应估计精度。

    采集分数组成:
    α(x; D_t) = w_info * α_info(x) + w_div * α_div(x) + w_bin * α_bin(x) + w_expl * α_expl(x)

    其中:
    - α_info: 信息增益（主效应和交互效应方差降低）
    - α_div: 多样性得分（空间覆盖度）
    - α_bin: 分区均匀性得分（因变量分区平衡）
    - α_expl: 探索奖励（适度exploration）
    - w_*: 动态调整的权重

    关键改进:
    1. 重复采样惩罚: 已采样设计得分设为-∞或大幅降低
    2. 强化多样性: 使用Gower距离 + 密度估计，对低覆盖区域提升权重
    3. 分区管理: 实时统计因变量分区采样率，优先采样稀疏区间
    4. 信息增益: 精确量化对主效应和交互效应后验方差的预期降低量
    5. 动态策略: 早期重exploration，后期重信息增益，自适应平衡
    6. 高分探索: UCB-like奖励，避免陷入局部最优

    Parameters
    ----------
    model : Model
        BoTorch/GPyTorch模型
    config_ini_path : str or Path, optional
        配置文件路径
    lambda_main : float, default=1.0
        主效应权重
    lambda_inter : float, default=0.5
        交互效应初始权重（会动态调整）
    lambda_inter_max : float, default=2.0
        交互效应最大权重
    tau_main : float, default=0.3
        主效应方差阈值（低于此值时增强交互效应）
    tau_inter : float, default=0.5
        交互效应方差阈值
    gamma_diversity : float, default=0.8
        多样性权重（早期）
    gamma_diversity_min : float, default=0.2
        多样性最小权重（后期）
    gamma_binning : float, default=0.3
        分区均匀性权重
    gamma_exploration : float, default=0.2
        探索奖励权重
    beta_ucb : float, default=0.1
        UCB exploration参数
    penalty_repeat : float, default=0.01
        重复采样惩罚系数（设为0.01意味着重复点得分降至1%）
    n_bins : int, default=5
        因变量分区数量
    coverage_method : str, default='min_distance'
        覆盖度计算方法
    interaction_terms : List[Tuple[int, int]], optional
        交互项列表
    noise_variance : float, default=1.0
        GP噪声方差
    prior_variance : float, default=1.0
        先验方差
    variable_types : Dict[int, str], optional
        变量类型（Gower距离用）
    """

    def __init__(
        self,
        model: Model,
        config_ini_path: Optional[Union[str, Path]] = None,
        # 信息增益相关
        lambda_main: float = 1.0,
        lambda_inter: float = 0.5,
        lambda_inter_max: float = 2.0,
        tau_main: float = 0.3,
        tau_inter: float = 0.5,
        # 多样性和覆盖相关
        gamma_diversity: float = 0.8,
        gamma_diversity_min: float = 0.2,
        gamma_binning: float = 0.3,
        gamma_exploration: float = 0.2,
        beta_ucb: float = 0.1,
        # 重复惩罚
        penalty_repeat: float = 0.01,
        # 分区设置
        n_bins: int = 5,
        coverage_method: str = "min_distance",
        # GP相关
        interaction_terms: Optional[List[Tuple[int, int]]] = None,
        noise_variance: float = 1.0,
        prior_variance: float = 1.0,
        variable_types: Optional[Dict[int, str]] = None,
    ):
        super().__init__(model=model)

        # 确保n_bins是整数（AEPsych可能将其作为float传入）
        n_bins = int(n_bins) if n_bins is not None else 5

        # 从配置文件加载参数（如果提供）
        if config_ini_path is not None:
            config = self._load_config(config_ini_path)
            lambda_main = config.get("lambda_main", lambda_main)
            lambda_inter = config.get("lambda_inter", lambda_inter)
            lambda_inter_max = config.get("lambda_inter_max", lambda_inter_max)
            tau_main = config.get("tau_main", tau_main)
            tau_inter = config.get("tau_inter", tau_inter)
            gamma_diversity = config.get("gamma_diversity", gamma_diversity)
            gamma_diversity_min = config.get("gamma_diversity_min", gamma_diversity_min)
            gamma_binning = config.get("gamma_binning", gamma_binning)
            gamma_exploration = config.get("gamma_exploration", gamma_exploration)
            beta_ucb = config.get("beta_ucb", beta_ucb)
            penalty_repeat = config.get("penalty_repeat", penalty_repeat)
            n_bins = config.get("n_bins", n_bins)
            coverage_method = config.get("coverage_method", coverage_method)
            interaction_terms = config.get("interaction_terms", interaction_terms)
            noise_variance = config.get("noise_variance", noise_variance)
            prior_variance = config.get("prior_variance", prior_variance)

        # 存储参数
        self.lambda_main = lambda_main
        self.lambda_inter = lambda_inter
        self.lambda_inter_max = lambda_inter_max
        self.tau_main = tau_main
        self.tau_inter = tau_inter
        self.gamma_diversity = gamma_diversity
        self.gamma_diversity_min = gamma_diversity_min
        self.gamma_binning = gamma_binning
        self.gamma_exploration = gamma_exploration
        self.beta_ucb = beta_ucb
        self.penalty_repeat = penalty_repeat
        self.n_bins = int(n_bins)  # 确保是整数
        self.coverage_method = coverage_method
        self.noise_variance = noise_variance
        self.prior_variance = prior_variance
        self.variable_types = variable_types

        # 解析交互项
        if isinstance(interaction_terms, str):
            self.interaction_terms = self._parse_interaction_terms(interaction_terms)
        else:
            self.interaction_terms = (
                interaction_terms if interaction_terms is not None else []
            )

        # 初始化GP计算器
        self.gp_calculator = GPVarianceCalculator(
            noise_variance=noise_variance,
            prior_variance=prior_variance,
            include_intercept=True,
        )

        # 内部状态
        self._X_train = None
        self._y_train = None
        self._var_initial = None
        self._var_current = None
        self._n_features = None
        self._fitted = False
        self._n_trials = 0  # 当前试验次数

        # 采样历史（用于重复检测）
        self._sampled_designs = set()  # 存储设计的字符串表示

        # 分区统计
        self._bin_counts = np.zeros(n_bins)  # 每个分区的样本数
        self._y_min = None
        self._y_max = None

    def _parse_interaction_terms(self, terms_str: str) -> List[Tuple[int, int]]:
        """解析交互项字符串"""
        if not terms_str or not terms_str.strip():
            return []

        terms = []
        for term in terms_str.split(";"):
            term = term.strip()
            if term:
                term = term.strip("()")
                parts = term.split(",")
                if len(parts) == 2:
                    try:
                        i, j = int(parts[0].strip()), int(parts[1].strip())
                        terms.append((i, j))
                    except ValueError:
                        pass
        return terms

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """从配置文件加载参数"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        parser = configparser.ConfigParser()
        parser.read(config_path, encoding="utf-8")

        config = {}

        # 尝试加载EnhancedVarianceReductionAcqf section
        section_name = None
        if "EnhancedVarianceReductionAcqf" in parser:
            section_name = "EnhancedVarianceReductionAcqf"
        elif "VarianceReductionWithCoverageAcqf" in parser:  # 兼容旧格式
            section_name = "VarianceReductionWithCoverageAcqf"

        if section_name is not None:
            section = parser[section_name]

            def parse_value(value_str):
                value_str = value_str.split("#")[0].strip()
                return value_str

            # 数值参数
            float_params = [
                "lambda_main",
                "lambda_inter",
                "lambda_inter_max",
                "tau_main",
                "tau_inter",
                "gamma_diversity",
                "gamma_diversity_min",
                "gamma_binning",
                "gamma_exploration",
                "beta_ucb",
                "penalty_repeat",
                "noise_variance",
                "prior_variance",
            ]
            for param in float_params:
                if param in section:
                    config[param] = float(parse_value(section[param]))

            # 整数参数
            if "n_bins" in section:
                config["n_bins"] = int(float(parse_value(section["n_bins"])))

            # 字符串参数
            if "coverage_method" in section:
                config["coverage_method"] = parse_value(section["coverage_method"])

            # 交互项
            if "interaction_terms" in section:
                terms_str = parse_value(section["interaction_terms"])
                if terms_str.strip():
                    terms = []
                    for term in terms_str.split(";"):
                        term = term.strip()
                        if term:
                            term = term.strip("()")
                            parts = term.split(",")
                            if len(parts) == 2:
                                try:
                                    i, j = int(parts[0].strip()), int(parts[1].strip())
                                    terms.append((i, j))
                                except ValueError:
                                    pass
                    config["interaction_terms"] = terms

        return config

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_types: Optional[Dict[int, str]] = None,
    ) -> "EnhancedVarianceReductionAcqf":
        """
        拟合采集函数

        Parameters
        ----------
        X : np.ndarray
            训练输入 (n_samples, n_features)
        y : np.ndarray
            训练输出 (n_samples,)
        variable_types : Dict[int, str], optional
            变量类型

        Returns
        -------
        self
        """
        if variable_types is not None:
            self.variable_types = variable_types

        self._X_train = X.copy()
        self._y_train = y.copy()
        self._n_features = X.shape[1]
        self._n_trials = len(y)

        # 更新采样历史（用于重复检测）
        self._sampled_designs.clear()
        for x in X:
            design_key = self._design_to_key(x)
            self._sampled_designs.add(design_key)

        # 更新分区统计
        if self._y_min is None:
            self._y_min = np.min(y)
            self._y_max = np.max(y)
        else:
            self._y_min = min(self._y_min, np.min(y))
            self._y_max = max(self._y_max, np.max(y))

        # 计算每个分区的样本数
        self._bin_counts = np.zeros(self.n_bins)
        if self._y_max > self._y_min:
            bin_edges = np.linspace(self._y_min, self._y_max, self.n_bins + 1)
            for y_val in y:
                bin_idx = np.digitize(y_val, bin_edges[1:-1])
                bin_idx = min(bin_idx, self.n_bins - 1)
                self._bin_counts[bin_idx] += 1

        # 拟合GP模型
        self.gp_calculator.fit(X, y, self.interaction_terms)

        # 存储初始方差（第一次拟合时）
        if self._var_initial is None:
            self._var_initial = self.gp_calculator.get_parameter_variance()

        # 存储当前方差
        self._var_current = self.gp_calculator.get_parameter_variance()

        self._fitted = True

        return self

    def _design_to_key(self, x: np.ndarray) -> str:
        """将设计点转换为字符串key（用于重复检测）"""
        # 对于分类变量，直接比较；对于连续变量，保留有限精度
        # 确保x是1维数组
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
        x_flat = x.flatten()
        return ",".join([f"{float(val):.6f}" for val in x_flat])

    def _compute_dynamic_weights(self) -> Tuple[float, float, float, float]:
        """
        动态计算各组件权重

        Returns
        -------
        w_info : float
            信息增益权重
        w_div : float
            多样性权重
        w_bin : float
            分区均匀性权重
        w_expl : float
            探索奖励权重
        """
        if not self._fitted:
            return 1.0, self.gamma_diversity, self.gamma_binning, self.gamma_exploration

        # 计算主效应方差降低比例
        offset = 1 if self.gp_calculator.include_intercept else 0
        n_main = self._n_features

        var_current_main = self._var_current[offset : offset + n_main]
        var_initial_main = self._var_initial[offset : offset + n_main]

        valid_mask = var_initial_main > 1e-10
        if np.any(valid_mask):
            r_main = np.mean(
                var_current_main[valid_mask] / var_initial_main[valid_mask]
            )
        else:
            r_main = 1.0

        # 计算交互效应方差降低比例
        if len(self.interaction_terms) > 0:
            n_inter = len(self.interaction_terms)
            var_current_inter = self._var_current[
                offset + n_main : offset + n_main + n_inter
            ]
            var_initial_inter = self._var_initial[
                offset + n_main : offset + n_main + n_inter
            ]

            valid_mask_inter = var_initial_inter > 1e-10
            if np.any(valid_mask_inter):
                r_inter = np.mean(
                    var_current_inter[valid_mask_inter]
                    / var_initial_inter[valid_mask_inter]
                )
            else:
                r_inter = 1.0
        else:
            r_inter = 1.0

        # 动态调整交互效应权重
        # 主效应收敛后（r_main < tau_main），增强交互效应权重
        if r_main < self.tau_main:
            lambda_inter_t = self.lambda_inter_max
        elif r_main > self.tau_inter:
            lambda_inter_t = self.lambda_inter
        else:
            # 线性插值
            lambda_inter_t = self.lambda_inter + (
                self.lambda_inter_max - self.lambda_inter
            ) * (self.tau_inter - r_main) / (self.tau_inter - self.tau_main)

        # 信息增益权重（固定为1，内部用lambda调整主/交互平衡）
        w_info = 1.0

        # 多样性权重随试验进行逐渐降低（早期exploration，后期exploitation）
        # 但保持一定下限以避免过度重复
        progress = self._n_trials / 80.0  # 假设总预算80次
        progress = min(progress, 1.0)
        w_div = (
            self.gamma_diversity
            - (self.gamma_diversity - self.gamma_diversity_min) * progress
        )

        # 分区均匀性权重保持固定
        w_bin = self.gamma_binning

        # 探索奖励权重（前期高，后期低）
        w_expl = self.gamma_exploration * (1.0 - progress)

        # 存储当前交互效应权重供信息增益计算使用
        self._current_lambda_inter = lambda_inter_t

        return w_info, w_div, w_bin, w_expl

    def _compute_info_gain(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        计算信息增益分数

        α_info(x) = λ_main * (1/|J|) Σ_j ΔVar[θ_j] + λ_inter * (1/|I|) Σ_{j,k} ΔVar[θ_jk]

        其中:
        - λ_main: 主效应权重
        - λ_inter: 交互效应权重（动态调整）
        - ΔVar: 预期方差降低量
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        n_candidates = X_candidates.shape[0]
        info_scores = np.zeros(n_candidates)

        # 使用动态调整的交互效应权重
        lambda_inter_t = getattr(self, "_current_lambda_inter", self.lambda_inter)

        for i in range(n_candidates):
            x = X_candidates[i : i + 1]

            # 计算方差降低
            main_var_red, inter_var_red = self.gp_calculator.compute_variance_reduction(
                x
            )

            # 主效应贡献
            avg_main_var_red = np.mean(main_var_red) if len(main_var_red) > 0 else 0.0

            # 交互效应贡献
            avg_inter_var_red = (
                np.mean(inter_var_red) if len(inter_var_red) > 0 else 0.0
            )

            # 组合（带权重）
            info_scores[i] = (
                self.lambda_main * avg_main_var_red + lambda_inter_t * avg_inter_var_red
            )

        return info_scores

    def _compute_diversity_score(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        计算多样性得分（空间覆盖度）

        使用Gower距离计算候选点到已采样点的最小距离
        距离越大，多样性越好
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # 使用Gower距离计算覆盖度
        diversity_scores = compute_coverage_batch(
            X_candidates,
            self._X_train,
            variable_types=self.variable_types,
            ranges=None,
            method=self.coverage_method,
        )

        return diversity_scores

    def _compute_binning_score(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        计算分区均匀性得分

        预测候选点的y值，判断其所属分区，对样本稀少的分区给予更高分数
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        n_candidates = X_candidates.shape[0]
        binning_scores = np.zeros(n_candidates)

        # 如果没有有效分区范围，返回0
        if self._y_min is None or self._y_max is None or self._y_max <= self._y_min:
            return binning_scores

        # 预测候选点的y值（使用GP均值）
        # 这里需要从GP模型获取预测
        # 确保是torch tensor
        if isinstance(X_candidates, torch.Tensor):
            X_tensor = X_candidates.float()
        else:
            X_tensor = torch.from_numpy(X_candidates).float()

        with torch.no_grad():
            try:
                posterior = self.model.posterior(X_tensor)
                y_pred = posterior.mean.cpu().numpy().flatten()
            except:
                # 如果预测失败，返回0分
                return binning_scores

        # 计算每个候选点的分区得分
        bin_edges = np.linspace(self._y_min, self._y_max, self.n_bins + 1)
        total_samples = np.sum(self._bin_counts)

        for i, y_val in enumerate(y_pred):
            # 确定所属分区
            bin_idx = np.digitize(y_val, bin_edges[1:-1])
            bin_idx = min(bin_idx, self.n_bins - 1)

            # 计算该分区的采样率
            if total_samples > 0:
                bin_rate = self._bin_counts[bin_idx] / total_samples
            else:
                bin_rate = 0.0

            # 采样率越低，得分越高（鼓励采样稀疏分区）
            # 使用反比例关系，加上小的平滑项避免除零
            binning_scores[i] = 1.0 / (bin_rate + 0.1)

        # 归一化到[0, 1]
        if np.max(binning_scores) > 0:
            binning_scores = binning_scores / np.max(binning_scores)

        return binning_scores

    def _compute_exploration_score(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        计算探索奖励得分（UCB-like）

        对预测不确定性高的区域给予奖励，鼓励exploration
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # 获取GP预测的不确定性（标准差）
        # 确保是torch tensor
        if isinstance(X_candidates, torch.Tensor):
            X_tensor = X_candidates.float()
            n_candidates = X_tensor.shape[0]
        else:
            X_tensor = torch.from_numpy(X_candidates).float()
            n_candidates = X_candidates.shape[0]

        with torch.no_grad():
            try:
                posterior = self.model.posterior(X_tensor)
                y_std = posterior.variance.sqrt().cpu().numpy().flatten()
            except:
                # 如果预测失败，返回0分
                return np.zeros(n_candidates)

        # UCB-like: 均值 + beta * 标准差
        # 这里只用标准差部分作为exploration bonus
        exploration_scores = self.beta_ucb * y_std

        # 归一化
        if np.max(exploration_scores) > 0:
            exploration_scores = exploration_scores / np.max(exploration_scores)

        return exploration_scores

    def _apply_repeat_penalty(
        self, X_candidates: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """
        对已采样的设计应用重复惩罚

        Parameters
        ----------
        X_candidates : np.ndarray
            候选点
        scores : np.ndarray
            原始得分

        Returns
        -------
        np.ndarray
            应用惩罚后的得分
        """
        penalized_scores = scores.copy()

        for i, x in enumerate(X_candidates):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                # 已采样过的设计，得分大幅降低
                penalized_scores[i] *= self.penalty_repeat

        return penalized_scores

    def _evaluate_numpy(self, X_candidates: np.ndarray) -> np.ndarray:
        """
        评估候选点的采集分数（内部numpy接口）

        Parameters
        ----------
        X_candidates : np.ndarray
            候选点 (n_candidates, n_features)

        Returns
        -------
        np.ndarray
            采集分数 (n_candidates,)
        """
        if X_candidates.ndim == 1:
            X_candidates = X_candidates.reshape(1, -1)

        # 计算动态权重
        w_info, w_div, w_bin, w_expl = self._compute_dynamic_weights()

        # 计算各组件得分
        info_scores = self._compute_info_gain(X_candidates)
        diversity_scores = self._compute_diversity_score(X_candidates)
        binning_scores = self._compute_binning_score(X_candidates)
        exploration_scores = self._compute_exploration_score(X_candidates)

        # 归一化各组件（使其在相似尺度）
        def safe_normalize(arr):
            if np.max(arr) > 0:
                return arr / np.max(arr)
            return arr

        info_scores = safe_normalize(info_scores)
        diversity_scores = safe_normalize(diversity_scores)
        binning_scores = safe_normalize(binning_scores)
        exploration_scores = safe_normalize(exploration_scores)

        # 加权组合
        total_scores = (
            w_info * info_scores
            + w_div * diversity_scores
            + w_bin * binning_scores
            + w_expl * exploration_scores
        )

        # 应用重复惩罚
        total_scores = self._apply_repeat_penalty(X_candidates, total_scores)

        return total_scores

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        BoTorch接口: 评估采集函数

        Parameters
        ----------
        X : torch.Tensor
            候选点, shape (batch_size, q, d) or (batch_size, d)

        Returns
        -------
        torch.Tensor
            采集分数, shape (batch_size,)
        """
        # 从模型提取训练数据
        if hasattr(self.model, "train_inputs") and hasattr(self.model, "train_targets"):
            if self._X_train is None:
                X_train_tensor = self.model.train_inputs[0]
                y_train_tensor = self.model.train_targets

                X_train = X_train_tensor.cpu().detach().numpy()
                y_train = y_train_tensor.cpu().detach().numpy()

                self.fit(X_train, y_train)

        if not self._fitted:
            return torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

        # 转换为numpy计算
        X_np = X.cpu().detach().numpy()

        if X_np.ndim == 3:  # batch_size x q x d
            batch_size, q, d = X_np.shape
            X_np = X_np.reshape(-1, d)

        # 计算得分
        scores = self._evaluate_numpy(X_np)

        # 转回torch
        scores_tensor = torch.from_numpy(scores).to(dtype=X.dtype, device=X.device)

        if X.ndim == 3:
            scores_tensor = scores_tensor.reshape(batch_size, q).sum(dim=-1)

        return scores_tensor

    def __call__(
        self, X_candidates: np.ndarray, return_components: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Numpy接口: 评估采集函数

        Parameters
        ----------
        X_candidates : np.ndarray
            候选点 (n_candidates, n_features)
        return_components : bool, default=False
            是否返回各组件得分

        Returns
        -------
        scores : np.ndarray
            总得分
        components : Dict[str, np.ndarray], optional
            各组件得分字典
        """
        # 从模型提取训练数据（如果还没拟合）
        if hasattr(self.model, "train_inputs") and self.model.train_inputs is not None:
            if not self._fitted:
                X_train_tensor = self.model.train_inputs[0]
                y_train_tensor = self.model.train_targets
                X_train = X_train_tensor.cpu().detach().numpy()
                y_train = y_train_tensor.cpu().detach().numpy()
                self.fit(X_train, y_train)
        elif not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # 处理输入类型（可能是tensor或numpy）
        if isinstance(X_candidates, torch.Tensor):
            X_np = X_candidates.cpu().detach().numpy()
        else:
            X_np = X_candidates

        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        # 使用numpy版本进行计算
        X_candidates = X_np

        # 计算各组件
        w_info, w_div, w_bin, w_expl = self._compute_dynamic_weights()

        info_scores = self._compute_info_gain(X_candidates)
        diversity_scores = self._compute_diversity_score(X_candidates)
        binning_scores = self._compute_binning_score(X_candidates)
        exploration_scores = self._compute_exploration_score(X_candidates)

        # 归一化
        def safe_normalize(arr):
            if np.max(arr) > 0:
                return arr / np.max(arr)
            return arr

        info_scores_norm = safe_normalize(info_scores)
        diversity_scores_norm = safe_normalize(diversity_scores)
        binning_scores_norm = safe_normalize(binning_scores)
        exploration_scores_norm = safe_normalize(exploration_scores)

        # 加权组合
        total_scores = (
            w_info * info_scores_norm
            + w_div * diversity_scores_norm
            + w_bin * binning_scores_norm
            + w_expl * exploration_scores_norm
        )

        # 重复惩罚
        total_scores = self._apply_repeat_penalty(X_candidates, total_scores)

        # 转换为torch tensor
        total_scores_tensor = torch.tensor(
            total_scores, dtype=torch.float32, requires_grad=False
        )

        if return_components:
            components = {
                "info": torch.tensor(info_scores_norm, dtype=torch.float32),
                "diversity": torch.tensor(diversity_scores_norm, dtype=torch.float32),
                "binning": torch.tensor(binning_scores_norm, dtype=torch.float32),
                "exploration": torch.tensor(
                    exploration_scores_norm, dtype=torch.float32
                ),
                "weights": {
                    "w_info": w_info,
                    "w_div": w_div,
                    "w_bin": w_bin,
                    "w_expl": w_expl,
                },
            }
            return total_scores_tensor, components

        return total_scores_tensor

    def select_next(
        self, X_candidates: np.ndarray, n_select: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        选择下一个采样点

        Parameters
        ----------
        X_candidates : np.ndarray
            候选点
        n_select : int, default=1
            选择数量

        Returns
        -------
        selected_X : np.ndarray
            选中的点
        selected_indices : np.ndarray
            选中点的索引
        """
        scores = self(X_candidates)
        scores_np = (
            scores.cpu().detach().numpy()
            if isinstance(scores, torch.Tensor)
            else scores
        )

        selected_indices = np.argsort(scores_np)[-n_select:][::-1]
        selected_X = X_candidates[selected_indices]

        return selected_X, selected_indices

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        获取诊断信息

        Returns
        -------
        Dict[str, Any]
            诊断信息字典
        """
        if not self._fitted:
            return {"fitted": False}

        w_info, w_div, w_bin, w_expl = self._compute_dynamic_weights()

        offset = 1 if self.gp_calculator.include_intercept else 0
        n_main = self._n_features

        var_current_main = self._var_current[offset : offset + n_main]
        var_initial_main = self._var_initial[offset : offset + n_main]

        valid_mask = var_initial_main > 1e-10
        if np.any(valid_mask):
            r_main = np.mean(
                var_current_main[valid_mask] / var_initial_main[valid_mask]
            )
        else:
            r_main = 1.0

        diagnostics = {
            "fitted": True,
            "n_trials": self._n_trials,
            "n_unique_designs": len(self._sampled_designs),
            "r_main": r_main,
            "lambda_inter_current": getattr(
                self, "_current_lambda_inter", self.lambda_inter
            ),
            "weights": {
                "w_info": w_info,
                "w_div": w_div,
                "w_bin": w_bin,
                "w_expl": w_expl,
            },
            "bin_counts": self._bin_counts.tolist(),
            "y_range": (self._y_min, self._y_max) if self._y_min is not None else None,
        }

        return diagnostics

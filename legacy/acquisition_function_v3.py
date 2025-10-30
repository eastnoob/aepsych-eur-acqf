"""
V3 Acquisition Function: V1 + Hard Exclusion
采集函数V3：基于V1的最小化改进，仅添加硬重复排除

核心原则：
- 保持V1的全部框架（信息增益 + 覆盖度）
- 只添加硬排除机制消除重复采样
- 不引入新的组件或复杂度

预期改进：
- 唯一设计数: 39 → 55-65 (+40-65%)
- 高分发现: 8 → 12-16 (+50-100%)
"""

import numpy as np
import torch
from scipy.stats import norm
from configparser import ConfigParser
import sys
import os

# 添加父目录到路径以导入V1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acquisition_function import VarianceReductionWithCoverageAcqf


class HardExclusionAcqf(VarianceReductionWithCoverageAcqf):
    """
    V3方案A: V1 + 硬重复排除

    改进点：
    1. 硬排除已采样设计 - 在forward()中直接过滤,确保torch.topk永远看不到已采样设计
    2. 其他全部继承V1

    配置参数：
    - 与V1完全相同
    - 无需额外参数
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sampled_designs = set()  # 存储已采样设计的指纹
        self._hard_exclusion_penalty = -1e10  # 极大负数惩罚

    # ------------------------
    # 内部工具：获取/应用参数空间规范化（与模型一致）
    # ------------------------
    def _get_param_transforms(self):
        """尝试从模型包装器中获取参数变换（用于将候选与历史对齐）。"""
        try:
            # ParameterTransformedModel 上有 transforms
            return getattr(self.model, "transforms", None)
        except Exception:
            return None

    def _canonicalize_array(self, X_np: np.ndarray) -> np.ndarray:
        """将一批设计点转换为与模型训练历史相同的规范化/离散化空间。

        - 对分类/整数维度执行 round（通过 ParameterTransforms.Categorical/Round）
        - 保持连续维度不变
        - 与 model.train_inputs 一致，确保键匹配
        """
        try:
            tf = self._get_param_transforms()
            if tf is None:
                return X_np

            import torch

            # 统一成 2D
            if X_np.ndim == 1:
                X_np = X_np.reshape(1, -1)

            X_t = torch.tensor(X_np, dtype=torch.float64)
            X_t_can = tf.transform(X_t)  # 会对分类/整数维度进行 round
            return X_t_can.detach().cpu().numpy()
        except Exception:
            # 变换不可用时保持原样
            return X_np

    def fit(self, X, y, variable_types=None):
        """拟合并记录采样历史"""
        super().fit(X, y, variable_types=variable_types)

        # ✅ 关键修复: 不清空,直接累积添加新设计
        # X 包含所有历史数据,直接全量更新即可
        self._sampled_designs.clear()  # 清空后重建是正确的,因为X已经是全量数据
        # 历史X来自 model.train_inputs（已在变换空间），再规范一次以保证一致
        X_can = self._canonicalize_array(X)
        for x in X_can:
            design_key = self._design_to_key(x)
            self._sampled_designs.add(design_key)

        print(
            f"[HardExclusionAcqf] 已记录 {len(self._sampled_designs)} 个唯一设计 (来自 {len(X)} 条历史数据)"
        )
        if len(self._sampled_designs) < len(X):
            print(
                f"[HardExclusionAcqf] ⚠️ 警告: {len(X) - len(self._sampled_designs)} 条重复采样!"
            )

    def _design_to_key(self, x):
        """将设计转换为唯一字符串标识"""
        if isinstance(x, np.ndarray):
            if x.ndim > 1:
                x = x.flatten()
            # 使用固定小数以避免微小的浮点误差导致键不一致
            return "_".join([f"{float(val):.6f}" for val in x])
        return str(x)

    def forward(self, X: "torch.Tensor") -> "torch.Tensor":
        """
        重写forward方法实现硬排除 - 这是BoTorch/AEPsych真正调用的方法!

        关键改进: 直接给已采样设计极大负数惩罚,确保torch.topk永远不会选中它们
        """
        import torch

        # ✅ 确保每次调用前都用最新训练数据刷新历史与内部状态（防止仅在首次fit后不再更新）
        try:
            if (
                hasattr(self, "model")
                and hasattr(self.model, "train_inputs")
                and self.model.train_inputs is not None
            ):
                X_train_tensor = self.model.train_inputs[0]
                y_train_tensor = self.model.train_targets
                if X_train_tensor is not None and y_train_tensor is not None:
                    X_train = X_train_tensor.cpu().detach().numpy()
                    y_train = y_train_tensor.cpu().detach().numpy()
                    # 始终刷新（代价可接受），保证_sampled_designs与GP状态最新
                    self.fit(X_train, y_train)
        except Exception as e:
            print(f"[HardExclusionAcqf] forward() 刷新历史失败: {e}")

        # 在进入父类计算前，先把候选点在参数空间中规范化（对分类/整数进行round）
        try:
            tf = self._get_param_transforms()
            if tf is not None:
                X = tf.transform(X)
        except Exception:
            pass

        # 先调用父类forward获取基础分数（V1逻辑）
        scores = super().forward(X)

        # 应用硬排除
        X_np = X.cpu().detach().numpy()
        if X_np.ndim == 3:  # batch_size x q x d
            X_np = X_np[:, 0, :]  # 取第一个点

        # 与历史一致地规范化候选（关键：确保键空间一致）
        X_np_can = self._canonicalize_array(X_np)

        excluded_count = 0
        for i, x in enumerate(X_np_can):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                scores[i] = self._hard_exclusion_penalty  # 极大负数
                excluded_count += 1

        if excluded_count > 0:
            print(
                f"[HardExclusionAcqf] forward()硬排除 {excluded_count}/{len(X_np)} 个已采样设计"
            )

        return scores

    def __call__(self, X_candidates, return_components=False):
        """
        重写__call__以实现硬排除逻辑

        **核心策略**: 给已采样设计设置极大负数惩罚,确保torch.topk永远不会选中
        """
        import torch

        # ✅ 在每次评分前，用模型中的最新训练数据刷新历史，确保硬排除覆盖到最新已采样点
        try:
            if (
                hasattr(self, "model")
                and hasattr(self.model, "train_inputs")
                and self.model.train_inputs is not None
            ):
                X_train_tensor = self.model.train_inputs[0]
                y_train_tensor = self.model.train_targets
                if X_train_tensor is not None and y_train_tensor is not None:
                    X_train = X_train_tensor.cpu().detach().numpy()
                    y_train = y_train_tensor.cpu().detach().numpy()
                    # 始终刷新（代价可接受），保证_sampled_designs与GP状态最新
                    self.fit(X_train, y_train)
        except Exception as e:
            print(f"[HardExclusionAcqf] __call__() 刷新历史失败: {e}")

        # 转换为numpy调用父类
        original_shape = X_candidates.shape
        if isinstance(X_candidates, torch.Tensor):
            is_3d = len(original_shape) == 3
            if is_3d:
                X_for_parent = X_candidates.squeeze(1).cpu().detach().numpy()
            else:
                X_for_parent = X_candidates.cpu().detach().numpy()
        else:
            is_3d = len(original_shape) == 3
            if is_3d:
                X_for_parent = X_candidates[:, 0, :]
            else:
                X_for_parent = X_candidates

        # 在评分前对候选进行规范化（分类/整数round），使之与历史同空间
        X_for_parent_can = self._canonicalize_array(X_for_parent)

        # 调用父类获取基础得分（避免向 forward 传递非预期参数）
        total_scores = super().__call__(X_for_parent_can)

        # 转换为torch tensor处理(保持类型一致性)
        if isinstance(total_scores, torch.Tensor):
            scores_tensor = total_scores.clone()
        else:
            scores_tensor = torch.tensor(total_scores, dtype=torch.float32)

        # 获取候选设计数组
        if is_3d:
            if isinstance(X_candidates, torch.Tensor):
                X_to_check = X_candidates[:, 0, :].cpu().detach().numpy()
            else:
                X_to_check = X_candidates[:, 0, :]
        else:
            if isinstance(X_candidates, torch.Tensor):
                X_to_check = X_candidates.cpu().detach().numpy()
            else:
                X_to_check = X_candidates

        # 对候选进行规范化以与历史一致
        X_to_check_can = self._canonicalize_array(X_to_check)

        # 硬排除已采样设计 - 直接设置为极大负数
        excluded_count = 0
        # 统计候选集中与历史命中的数量（用于诊断键一致性）
        try:
            candidate_keys = set()
            for x in X_to_check_can[: min(len(X_to_check_can), 10000)]:  # 安全上限
                candidate_keys.add(self._design_to_key(x))
            hit_count = len(candidate_keys & self._sampled_designs)
        except Exception:
            hit_count = -1
        for i, x in enumerate(X_to_check_can):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                scores_tensor[i] = self._hard_exclusion_penalty  # -1e10
                excluded_count += 1

        if excluded_count > 0 or hit_count == 0:
            print(
                f"[HardExclusionAcqf] __call__()硬排除 {excluded_count}/{len(X_to_check)} 个; 候选命中历史={hit_count}, 历史唯一={len(self._sampled_designs)}"
            )

        if return_components:
            # 兼容返回签名，但此处不提供组件分解（仅返回 None 占位）
            return scores_tensor, None, None
        else:
            return scores_tensor

    def select_next(self, X_candidates, n_select=1):
        """
        选择下一个采样点，确保不重复

        关键改进：跳过已采样设计，选择未采样的top-n
        确保返回的数量 == n_select（即使需要选次优）
        """
        # 获取所有得分
        scores = self(X_candidates)

        # 标识未采样的设计
        unsampled_mask = np.ones(len(X_candidates), dtype=bool)
        for i, x in enumerate(X_candidates):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                unsampled_mask[i] = False

        # 获取未采样设计的索引
        unsampled_indices = np.where(unsampled_mask)[0]

        if len(unsampled_indices) == 0:
            # 如果所有候选都已采样（极端情况），随机选择
            print(f"[HardExclusionAcqf] 警告：所有候选设计都已采样，随机选择")
            selected_indices = np.random.choice(
                len(X_candidates), size=min(n_select, len(X_candidates)), replace=False
            )
        else:
            # 获取未采样设计的得分
            unsampled_scores = scores[unsampled_indices]

            # 按得分降序排列（使用argsort，然后反转）
            sorted_order = np.argsort(-unsampled_scores)  # 负号实现降序
            sorted_unsampled_idx = unsampled_indices[sorted_order]

            # 选择top-n
            n_available = min(n_select, len(sorted_unsampled_idx))
            selected_indices = sorted_unsampled_idx[:n_available]

            if n_available < n_select:
                print(
                    f"[HardExclusionAcqf] 警告：仅有{n_available}个未采样设计，少于请求的{n_select}个"
                )

        selected_X = X_candidates[selected_indices]

        return selected_X, selected_indices


class CombinedAcqf(VarianceReductionWithCoverageAcqf):
    """
    V3方案C: V1 + 候选集预过滤 + 硬排除

    改进点：
    1. 候选集生成时优先未采样设计（80%未采样 + 20%已采样）
    2. 评分时仍然硬排除已采样设计（双重保险）
    3. 其他全部继承V1

    配置参数：
    - 与V1完全相同
    - candidate_unsampled_ratio: 候选集中未采样设计的比例（默认0.8）
    """

    def __init__(self, candidate_unsampled_ratio=0.8, **kwargs):
        super().__init__(**kwargs)
        self._sampled_designs = set()
        self.candidate_unsampled_ratio = candidate_unsampled_ratio

    def __call__(self, X_candidates, return_components=False):
        """
        重写__call__以实现硬排除逻辑

        关键修改：在最终得分中对已采样设计设置-inf
        **核心修复**: 必须转换为numpy再调用父类,因为V1期望np.ndarray
        """
        import torch

        # X_candidates可能是[batch, q, dim]或[batch, dim]
        # 需要处理q维度,并转换为numpy给父类
        original_shape = X_candidates.shape
        if isinstance(X_candidates, torch.Tensor):
            is_3d = len(original_shape) == 3
            if is_3d:
                # 移除q维度 [batch, 1, dim] -> [batch, dim]
                X_for_parent = X_candidates.squeeze(1).cpu().detach().numpy()
            else:
                X_for_parent = X_candidates.cpu().detach().numpy()
        else:
            is_3d = len(original_shape) == 3
            if is_3d:
                X_for_parent = X_candidates[:, 0, :]
            else:
                X_for_parent = X_candidates

        # 调用父类获取基础得分(父类期望numpy输入,返回torch.Tensor)；避免传递非预期参数
        total_scores = super().__call__(X_for_parent)

        # 确保total_scores是numpy数组
        if isinstance(total_scores, torch.Tensor):
            scores_np = total_scores.cpu().detach().numpy().copy()
            was_tensor = True
        else:
            scores_np = np.array(total_scores).copy()
            was_tensor = False

        # 硬排除已采样设计
        excluded_count = 0
        # 处理X_candidates维度
        if is_3d:
            if isinstance(X_candidates, torch.Tensor):
                X_to_check = X_candidates[:, 0, :].cpu().detach().numpy()
            else:
                X_to_check = X_candidates[:, 0, :]
        else:
            if isinstance(X_candidates, torch.Tensor):
                X_to_check = X_candidates.cpu().detach().numpy()
            else:
                X_to_check = X_candidates

        for i, x in enumerate(X_to_check):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                scores_np[i] = -np.inf  # 完全排除
                excluded_count += 1

        if excluded_count > 0:
            print(
                f"[CombinedAcqf] 本轮硬排除 {excluded_count}/{len(X_to_check)} 个已采样设计"
            )

        # 转回tensor（如果原本是tensor）
        if was_tensor:
            total_scores = torch.tensor(
                scores_np, dtype=torch.float32, requires_grad=False
            )
        else:
            total_scores = scores_np

        if return_components:
            return total_scores, None, None
        else:
            return total_scores

    def fit(self, X, y, variable_types=None):
        """拟合并记录采样历史"""
        super().fit(X, y, variable_types=variable_types)

        # 清空并重新记录所有已采样设计
        self._sampled_designs.clear()
        for x in X:
            design_key = self._design_to_key(x)
            self._sampled_designs.add(design_key)

        print(f"[CombinedAcqf] 已记录 {len(self._sampled_designs)} 个唯一设计")

    def _design_to_key(self, x):
        """将设计转换为唯一字符串标识"""
        if isinstance(x, np.ndarray):
            if x.ndim > 1:
                x = x.flatten()
            return "_".join([f"{float(val):.6f}" for val in x])
        return str(x)

    def filter_candidates(self, X_candidates):
        """
        过滤候选集，优先未采样设计

        Args:
            X_candidates: 所有候选设计 (n_candidates, n_features)

        Returns:
            filtered: 过滤后的候选集，优先包含未采样设计
        """
        if len(self._sampled_designs) == 0:
            return X_candidates

        # 分类候选集
        unsampled_idx = []
        sampled_idx = []

        for i, x in enumerate(X_candidates):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                sampled_idx.append(i)
            else:
                unsampled_idx.append(i)

        # 计算目标数量
        n_total = len(X_candidates)
        n_unsampled_target = int(n_total * self.candidate_unsampled_ratio)
        n_sampled_target = n_total - n_unsampled_target

        # 采样
        n_unsampled = min(len(unsampled_idx), n_unsampled_target)
        n_sampled = min(len(sampled_idx), n_sampled_target)

        # 如果未采样设计不足，用已采样设计补充
        if n_unsampled < n_unsampled_target:
            n_sampled = min(len(sampled_idx), n_total - n_unsampled)

        selected_idx = []
        if n_unsampled > 0:
            selected_idx.extend(
                np.random.choice(unsampled_idx, size=n_unsampled, replace=False)
            )
        if n_sampled > 0:
            selected_idx.extend(
                np.random.choice(sampled_idx, size=n_sampled, replace=False)
            )

        return X_candidates[selected_idx]

    def _evaluate_numpy(self, X_candidates):
        """评估候选点，硬排除已采样设计"""
        # 使用V1的评分逻辑
        scores = super()._evaluate_numpy(X_candidates)

        # 硬排除已采样设计（双重保险）
        for i, x in enumerate(X_candidates):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                scores[i] = -np.inf

        return scores

    def select_next(self, X_candidates, n_select=1):
        """
        选择下一个采样点，确保不重复

        关键改进：跳过已采样设计，选择未采样的top-n
        确保返回的数量 == n_select（即使需要选次优）
        """
        # 获取所有得分
        scores = self(X_candidates)

        # 标识未采样的设计
        unsampled_mask = np.ones(len(X_candidates), dtype=bool)
        for i, x in enumerate(X_candidates):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                unsampled_mask[i] = False

        # 获取未采样设计的索引
        unsampled_indices = np.where(unsampled_mask)[0]

        if len(unsampled_indices) == 0:
            # 如果所有候选都已采样（极端情况），随机选择
            print(f"[CombinedAcqf] 警告：所有候选设计都已采样，随机选择")
            selected_indices = np.random.choice(
                len(X_candidates), size=min(n_select, len(X_candidates)), replace=False
            )
        else:
            # 获取未采样设计的得分
            unsampled_scores = scores[unsampled_indices]

            # 按得分降序排列（使用argsort，然后反转）
            sorted_order = np.argsort(-unsampled_scores)  # 负号实现降序
            sorted_unsampled_idx = unsampled_indices[sorted_order]

            # 选择top-n
            n_available = min(n_select, len(sorted_unsampled_idx))
            selected_indices = sorted_unsampled_idx[:n_available]

            if n_available < n_select:
                print(
                    f"[CombinedAcqf] 警告：仅有{n_available}个未采样设计，少于请求的{n_select}个"
                )

        selected_X = X_candidates[selected_indices]

        return selected_X, selected_indices


# 配置加载支持
def load_from_config(config_str, acqf_type="hard_exclusion"):
    """
    从配置字符串加载V3采集函数

    Args:
        config_str: INI格式配置字符串
        acqf_type: "hard_exclusion" 或 "combined"

    Returns:
        采集函数实例
    """
    config = ConfigParser()
    config.read_string(config_str)

    # V3使用与V1相同的配置section
    section = "VarianceReductionWithCoverageAcqf"
    if not config.has_section(section):
        section = list(config.sections())[-1]  # 使用最后一个section

    # 解析参数 (使用V1的参数名)
    params = {}

    # 基础参数
    if config.has_option(section, "lambda_min"):
        params["lambda_min"] = config.getfloat(section, "lambda_min")
    if config.has_option(section, "lambda_max"):
        params["lambda_max"] = config.getfloat(section, "lambda_max")
    if config.has_option(section, "tau_1"):
        params["tau_1"] = config.getfloat(section, "tau_1")
    if config.has_option(section, "tau_2"):
        params["tau_2"] = config.getfloat(section, "tau_2")
    if config.has_option(section, "gamma"):
        params["gamma"] = config.getfloat(section, "gamma")

    # 覆盖度方法
    if config.has_option(section, "coverage_method"):
        params["coverage_method"] = config.get(section, "coverage_method")

    # 交互项
    if config.has_option(section, "interaction_terms"):
        interaction_str = config.get(section, "interaction_terms")
        params["interaction_terms"] = []
        for term in interaction_str.split(";"):
            term = term.strip()
            if term:
                indices = term.strip("()").split(",")
                params["interaction_terms"].append(
                    (int(indices[0].strip()), int(indices[1].strip()))
                )

    # 组合方案的额外参数
    if acqf_type == "combined":
        if config.has_option(section, "candidate_unsampled_ratio"):
            params["candidate_unsampled_ratio"] = config.getfloat(
                section, "candidate_unsampled_ratio"
            )

    # 创建实例
    if acqf_type == "hard_exclusion":
        return HardExclusionAcqf(**params)
    elif acqf_type == "combined":
        return CombinedAcqf(**params)
    else:
        raise ValueError(f"Unknown acqf_type: {acqf_type}")


if __name__ == "__main__":
    # 简单测试
    print("V3 Acquisition Functions - 测试不重复选择逻辑")
    print("=" * 70)

    # 测试硬排除
    print("\n方案A: 硬排除")
    # V1参数名: lambda_min, lambda_max, tau_1, tau_2, gamma
    acqf_a = HardExclusionAcqf(
        model=None,  # 测试时不需要实际模型
        lambda_min=0.5,
        lambda_max=3.0,
        tau_1=0.5,
        tau_2=0.3,
        gamma=0.5,
        interaction_terms=[(0, 1), (1, 2)],
    )

    # 模拟数据
    X_train = np.array([[0, 1, 2], [1, 0, 1], [2, 2, 0]])
    y_train = np.array([0.8, 0.6, 0.7])
    acqf_a.fit(X_train, y_train)

    print(f"已采样设计: {acqf_a._sampled_designs}")

    # 测试1: select_next是否跳过已采样设计
    print("\n【测试1: select_next跳过已采样设计】")
    X_candidates = np.array(
        [
            [0, 1, 2],  # 已采样 (应该跳过)
            [1, 0, 1],  # 已采样 (应该跳过)
            [2, 2, 0],  # 已采样 (应该跳过)
            [3, 3, 3],  # 未采样
            [4, 4, 4],  # 未采样
            [5, 5, 5],  # 未采样
        ]
    )

    selected_X, selected_idx = acqf_a.select_next(X_candidates, n_select=2)
    print(f"请求选择: 2个设计")
    print(f"选中的索引: {selected_idx}")
    print(f"选中的设计: {selected_X.tolist()}")

    # 验证是否未采样
    all_unsampled = True
    for x in selected_X:
        key = acqf_a._design_to_key(x)
        if key in acqf_a._sampled_designs:
            all_unsampled = False
            print(f"❌ 错误：选中了已采样设计 {x.tolist()}")

    if all_unsampled:
        print(f"✅ 正确：所有选中的设计都是未采样的")

    # 测试2: 当候选集中有很多已采样设计时
    print("\n【测试2: 候选集中80%已采样】")
    X_many_sampled = np.array(
        [
            [0, 1, 2],
            [1, 0, 1],
            [2, 2, 0],  # 前3个已采样
            [0, 1, 2],
            [1, 0, 1],
            [2, 2, 0],  # 重复的已采样
            [0, 1, 2],
            [1, 0, 1],  # 又是已采样
            [3, 3, 3],
            [4, 4, 4],  # 最后2个未采样
        ]
    )

    selected_X2, selected_idx2 = acqf_a.select_next(X_many_sampled, n_select=2)
    print(f"候选集: 10个设计，其中8个已采样，2个未采样")
    print(f"选中的索引: {selected_idx2}")
    print(f"选中的设计: {selected_X2.tolist()}")

    # 验证
    all_unsampled2 = True
    for x in selected_X2:
        key = acqf_a._design_to_key(x)
        if key in acqf_a._sampled_designs:
            all_unsampled2 = False
            print(f"❌ 错误：选中了已采样设计 {x.tolist()}")

    if all_unsampled2:
        print(f"✅ 正确：即使候选集80%已采样，仍然只选择了未采样设计")

    # 测试组合方案
    print("\n" + "=" * 70)
    print("方案C: 组合方案")
    acqf_c = CombinedAcqf(
        model=None,
        lambda_min=0.5,
        lambda_max=3.0,
        tau_1=0.5,
        tau_2=0.3,
        gamma=0.5,
        candidate_unsampled_ratio=0.8,
        interaction_terms=[(0, 1), (1, 2)],
    )
    acqf_c.fit(X_train, y_train)

    # 测试候选集过滤 + select_next
    print("\n【测试3: 组合方案的候选集过滤】")
    filtered = acqf_c.filter_candidates(X_many_sampled)
    print(f"原始候选集大小: {len(X_many_sampled)}")
    print(f"过滤后候选集大小: {len(filtered)}")

    selected_X3, selected_idx3 = acqf_c.select_next(X_many_sampled, n_select=2)
    print(f"选中的设计: {selected_X3.tolist()}")

    # 验证
    all_unsampled3 = True
    for x in selected_X3:
        key = acqf_c._design_to_key(x)
        if key in acqf_c._sampled_designs:
            all_unsampled3 = False
            print(f"❌ 错误：选中了已采样设计 {x.tolist()}")

    if all_unsampled3:
        print(f"✅ 正确：组合方案也正确跳过已采样设计")

    print("\n" + "=" * 70)
    print("✅ V3不重复选择逻辑测试完成")
    print("=" * 70)

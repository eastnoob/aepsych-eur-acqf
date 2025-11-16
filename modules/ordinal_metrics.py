"""
序数响应模型信息度量模块

处理序数似然（OrdinalLikelihood）模型的熵计算和CDF评估。
适用于Likert量表等有序分类响应。

核心功能：
1. 稳定的序数熵计算（避免NaN/Inf）
2. cutpoints提取与验证
3. 模型类型检测与配置检查

Example:
    >>> helper = OrdinalMetricsHelper(model)
    >>> entropy = helper.compute_entropy(X_candidates)
"""

from __future__ import annotations
from typing import Optional
import warnings
import torch
from botorch.models.model import Model


EPS = 1e-8


class OrdinalMetricsHelper:
    """序数模型信息度量辅助类

    封装序数模型的熵计算逻辑，提供稳定的数值计算。
    """

    def __init__(self, model: Model):
        """
        Args:
            model: BoTorch模型（应包含OrdinalLikelihood）
        """
        self.model = model
        self._is_ordinal_cached: Optional[bool] = None
        self._cutpoints_cache: Optional[torch.Tensor] = None

    def is_ordinal(self) -> bool:
        """检测模型是否使用序数似然

        检查条件：
        1. likelihood类名包含"ordinal"
        2. 有n_levels属性
        3. 能提取到有效的cutpoints

        Returns:
            True if 序数模型
        """
        if self._is_ordinal_cached is not None:
            return self._is_ordinal_cached

        try:
            lk = getattr(self.model, "likelihood", None)
            if lk is None:
                self._is_ordinal_cached = False
                return False

            name = type(lk).__name__.lower()
            if "ordinal" in name:
                self._is_ordinal_cached = True
                return True

            if hasattr(lk, "n_levels") or hasattr(lk, "num_levels"):
                self._is_ordinal_cached = True
                return True

            # 尝试提取cutpoints作为最终检查
            c = self.get_cutpoints(device=torch.device("cpu"), dtype=torch.float64)
            self._is_ordinal_cached = (c is not None)
            return self._is_ordinal_cached

        except Exception:
            self._is_ordinal_cached = False
            return False

    def get_cutpoints(
        self,
        device: torch.device,
        dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """从likelihood提取cutpoints

        尝试多种可能的属性名：
        - cutpoints
        - cut_points
        - cut_points_
        - _cutpoints

        Args:
            device: 目标设备
            dtype: 目标数据类型

        Returns:
            (K,) 张量，K为cutpoint数量
            None if 无法提取
        """
        if not self.is_ordinal():
            return None

        # 使用缓存（设备/类型匹配时）
        if (self._cutpoints_cache is not None and
            self._cutpoints_cache.device == device and
            self._cutpoints_cache.dtype == dtype):
            return self._cutpoints_cache

        cand_names = ["cutpoints", "cut_points", "cut_points_", "_cutpoints"]
        lk = self.model.likelihood

        for name in cand_names:
            c = getattr(lk, name, None)
            if c is not None:
                try:
                    c_t = torch.as_tensor(c, device=device, dtype=dtype)
                    c_t = c_t.view(-1)  # 展平为1D
                    self._cutpoints_cache = c_t
                    return c_t
                except Exception:
                    continue

        return None

    @staticmethod
    def normal_cdf(z: torch.Tensor) -> torch.Tensor:
        """标准正态分布CDF

        Φ(z) = 0.5 * (1 + erf(z / √2))

        Args:
            z: 标准化值

        Returns:
            CDF值，范围[0, 1]
        """
        return 0.5 * (
            1.0 + torch.erf(
                z / torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype))
            )
        )

    def compute_ordinal_entropy(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        cutpoints: torch.Tensor
    ) -> torch.Tensor:
        """数值稳定的序数熵计算

        给定潜变量的后验分布 f ~ N(mean, var)，
        计算离散观测概率分布的熵：

        H = -∑ p_k * log(p_k)

        其中 p_k = P(y=k | mean, var) 通过cutpoints和正态CDF计算。

        数值稳定性改进：
        1. 严格的EPS边界（1e-7）
        2. 显式概率规范化（确保和为1）
        3. NaN/Inf安全检查

        Args:
            mean: (N,) 后验均值
            var: (N,) 后验方差
            cutpoints: (K,) cutpoints（已排序）

        Returns:
            (N,) 熵值
        """
        # 标准差（数值稳定）
        std = torch.sqrt(torch.clamp(var, min=EPS))

        # 标准化cutpoints: z = (c - μ) / σ
        z = (cutpoints.view(1, -1) - mean.view(-1, 1)) / std.view(-1, 1)

        # CDF值（夹值避免极端值）
        cdfs = self.normal_cdf(z).clamp(1e-7, 1 - 1e-7)

        # 计算每个类别的概率
        # p_0 = Φ(c_0)
        # p_k = Φ(c_k) - Φ(c_{k-1}), k=1,...,K-1
        # p_K = 1 - Φ(c_{K-1})

        p0 = cdfs[:, :1]
        p_last = 1.0 - cdfs[:, -1:]

        if cdfs.shape[1] >= 2:
            mids = torch.clamp(cdfs[:, 1:] - cdfs[:, :-1], min=1e-7)
            probs = torch.cat([p0, mids, p_last], dim=1)
        else:
            probs = torch.cat([p0, p_last], dim=1)

        # 【关键】显式规范化确保和为1
        probs = probs / (probs.sum(dim=1, keepdim=True) + EPS)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        # 【关键】数值稳定的熵计算
        log_probs = torch.log(torch.clamp(probs, min=1e-7))
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # 【关键】NaN/Inf安全检查（降级到均匀分布熵）
        n_levels = probs.shape[1]
        fallback_entropy = torch.log(torch.tensor(
            float(n_levels), device=entropy.device, dtype=entropy.dtype
        ))

        entropy = torch.where(
            torch.isnan(entropy) | torch.isinf(entropy),
            fallback_entropy.expand_as(entropy),
            entropy
        )

        return entropy

    def compute_entropy(self, X: torch.Tensor) -> torch.Tensor:
        """计算候选点的信息熵（高层接口）

        Args:
            X: (N, d) 候选点

        Returns:
            (N,) 熵值

        Raises:
            RuntimeError: 序数模型缺少cutpoints
        """
        if not self.is_ordinal():
            raise RuntimeError("Model is not ordinal, cannot compute entropy")

        with torch.no_grad():
            posterior = self.model.posterior(X)
            mean = posterior.mean
            var = getattr(posterior, "variance", None)

            # 维度归约
            def _reduce_event(x: torch.Tensor) -> torch.Tensor:
                while x.dim() > 1 and x.shape[-1] == 1:
                    x = x.squeeze(-1)
                if x.dim() > 1:
                    x = x.mean(dim=-1)
                return x.view(-1)

            mean_r = _reduce_event(mean)

            if var is None:
                base_var = torch.ones_like(mean_r)
            else:
                base_var = _reduce_event(var)

            cutpoints = self.get_cutpoints(
                device=mean_r.device,
                dtype=mean_r.dtype
            )

            if cutpoints is None:
                raise RuntimeError(
                    "Cannot extract cutpoints from ordinal model. "
                    "Ensure model is trained and cutpoints are accessible."
                )

            return self.compute_ordinal_entropy(mean_r, base_var, cutpoints)

    def check_config(self) -> None:
        """检查序数模型配置（用于初始化诊断）

        如果模型看起来像序数模型但无法获取cutpoints，
        发出警告提示用户检查配置。

        应在首次数据同步时调用（避免重复警告）。
        """
        if self.model is None:
            return

        likelihood = getattr(self.model, "likelihood", None)
        if likelihood is None:
            return

        # 检查是否"看起来像"序数似然
        likelihood_name = type(likelihood).__name__.lower()
        is_ordinal_like = (
            "ordinal" in likelihood_name
            or hasattr(likelihood, "n_levels")
            or hasattr(likelihood, "num_levels")
        )

        # 如果看起来像序数模型，检查是否有cutpoints属性
        if is_ordinal_like:
            cand_names = ["cutpoints", "cut_points", "cut_points_", "_cutpoints"]
            has_cutpoints = False

            for name in cand_names:
                attr = getattr(likelihood, name, None)
                if attr is not None:
                    try:
                        if torch.is_tensor(attr) and attr.numel() > 0:
                            has_cutpoints = True
                            break
                        elif hasattr(attr, "__len__") and len(attr) > 0:
                            has_cutpoints = True
                            break
                    except Exception:
                        continue

            # 如果看起来像序数模型但没有cutpoints，发出警告
            if not has_cutpoints:
                warnings.warn(
                    f"检测到序数似然模型（{type(likelihood).__name__}），但无法获取cutpoints。\n"
                    f"这可能表示配置错误。信息增益计算将退化为方差指标（Var[p̂]）。\n"
                    f"建议检查：\n"
                    f"  1. OrdinalLikelihood是否正确初始化（需要n_levels参数）\n"
                    f"  2. 模型是否已经过训练（cutpoints在训练时学习）\n"
                    f"  3. cutpoints属性是否可访问\n"
                    f"当前配置下将使用方差指标继续计算。",
                    UserWarning,
                )

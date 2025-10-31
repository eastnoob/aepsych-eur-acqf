"""
EUR Acquisition Function (V4): forward-only, fresh Dt, space-canonicalized, variable types aware.

本版本在 V1 的基础上，完成以下工程修正：
- 仅通过 forward 调用（不覆写 __call__），遵循 BoTorch 调用链与 t_batch 语义；
- 每次 forward 使用最新 Dt（必要时刷新内部状态与方差基线/当前值）；
- 评分前将候选与历史统一到与模型一致的“变换空间规范形”（分类/整数 round）；
- 支持从 ParameterTransforms 推断 variable_types（未显式提供时）。

目标：降低重复采样，严格符合 α(x;Dt)=α_info+α_cov 与动态 λ_t 设计。

这个函数与ordinal的兼容有问题，V5将对此进行解决。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import re

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform

# 相对/直接导入（与 V1 保持一致）
try:
    from .gower_distance import compute_coverage_batch
    from .gp_variance import GPVarianceCalculator
except Exception:  # pragma: no cover
    from gower_distance import compute_coverage_batch
    from gp_variance import GPVarianceCalculator


class EURAcqfV4(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        config_ini_path: Optional[str] = None,  # 兼容参数位
        lambda_min: float = 0.2,
        lambda_max: float = 2.0,
        tau_1: float = 0.5,
        tau_2: float = 0.1,
        gamma: float = 0.3,
        interaction_terms: Optional[List[Tuple[int, int]]] = None,
        noise_variance: float = 1.0,
        prior_variance: float = 1.0,
        variable_types: Optional[Dict[int, str]] = None,
        coverage_method: str = "min_distance",
        # 允许从配置传入按维度顺序的变量类型列表，例如:
        # ["categorical", "categorical", "integer", "continuous"]
        variable_types_list: Optional[Union[List[str], str]] = None,
        # 调试：记录并打印 info/cov 组成分值的统计
        debug_components: Union[bool, str] = False,
    ) -> None:
        super().__init__(model=model)

        # 超参（与 V1 一致）
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.gamma = gamma
        self.coverage_method = coverage_method

        # 交互项解析：支持多种分隔方式
        # 示例："(0,1);(1,2)" 或 "(0,1),(1,2)" 或 "[(0,1), (1,2)]"
        if isinstance(interaction_terms, str):
            s = interaction_terms.strip()
            # 直接用正则提取所有形如 (i,j) 的片段，鲁棒处理分号/逗号/空格等分隔
            pairs = re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", s)
            parsed: List[Tuple[int, int]] = [(int(i), int(j)) for i, j in pairs]
            self.interaction_terms = parsed
        else:
            self.interaction_terms = (
                interaction_terms if interaction_terms is not None else []
            )
        self.noise_variance = noise_variance
        self.prior_variance = prior_variance

        # GP 计算器
        self.gp_calculator = GPVarianceCalculator(
            noise_variance=noise_variance,
            prior_variance=prior_variance,
            include_intercept=True,
        )

        # 运行时缓存
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._var_initial: Optional[np.ndarray] = None
        self._var_current: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None
        self._fitted: bool = False
        self._last_hist_n: int = -1

        # 覆盖类型（可外部显式指定，或在首次可用时推断）
        self.variable_types: Optional[Dict[int, str]] = variable_types

        # 从列表规格构造 variable_types（优先级高于推断）
        # 允许从字符串/列表构建 variable_types
        if variable_types_list is not None and self.variable_types is None:
            if isinstance(variable_types_list, str):
                # 去掉括号/方括号，按逗号或分号拆分
                s = variable_types_list.strip()
                if (s.startswith("[") and s.endswith("]")) or (
                    s.startswith("(") and s.endswith(")")
                ):
                    s = s[1:-1]
                tokens = [
                    t.strip() for t in s.replace(";", ",").split(",") if t.strip()
                ]
            else:
                tokens = list(variable_types_list)
            vt_map: Dict[int, str] = {}
            for i, t in enumerate(tokens):
                if t is None:
                    continue
                t_l = str(t).lower()
                if t_l.startswith("cat"):
                    vt_map[i] = "categorical"
                elif t_l.startswith("int"):
                    vt_map[i] = "integer"
                # 连续维不需要特别标注
            if len(vt_map) > 0:
                self.variable_types = vt_map

        # 调试组件输出（将字符串解析为布尔）
        if isinstance(debug_components, str):
            self.debug_components = debug_components.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            self.debug_components = bool(debug_components)
        self._last_info: Optional[np.ndarray] = None
        self._last_cov: Optional[np.ndarray] = None

    # -------------------- 工具：获取并使用 ParameterTransforms --------------------
    def _get_param_transforms(self):
        try:
            return getattr(self.model, "transforms", None)
        except Exception:
            return None

    def _canonicalize_numpy(self, X: np.ndarray) -> np.ndarray:
        """将 numpy 数组按模型的参数变换规范化：分类/整数维 round，一致于训练空间。"""
        try:
            tf = self._get_param_transforms()
            if tf is None:
                return X
            X_t = torch.tensor(X, dtype=torch.float64)
            X_t_can = tf.transform(X_t)
            return X_t_can.detach().cpu().numpy()
        except Exception:
            return X

    def _maybe_infer_variable_types(self) -> None:
        """从 transforms 推断 variable_types（分类/整数）一次。"""
        if self.variable_types is not None:
            return
        tf = self._get_param_transforms()
        if tf is None:
            return
        vt: Dict[int, str] = {}
        try:
            # 这些类型来自 temp_aepsych.transforms.ops
            from aepsych.transforms.ops import Categorical, Round

            for sub in tf.values():
                if hasattr(sub, "indices") and isinstance(sub.indices, list):
                    for idx in sub.indices:
                        if isinstance(sub, Categorical):
                            vt[idx] = "categorical"
                        elif isinstance(sub, Round):
                            # 可按需要区分 integer；Gower 中可与连续处理不同
                            vt.setdefault(idx, "integer")
        except Exception:
            pass
        if len(vt) > 0:
            self.variable_types = vt

    # -------------------- EUR 计算核心 --------------------
    def _fit_internal(self, X: np.ndarray, y: np.ndarray) -> None:
        # 规范化历史
        X_can = self._canonicalize_numpy(X)

        self._X_train = X_can.copy()
        self._y_train = y.copy()
        self._n_features = X_can.shape[1]

        # 拟合 GP 方差计算器
        self.gp_calculator.fit(X_can, y, self.interaction_terms)

        if self._var_initial is None:
            self._var_initial = self.gp_calculator.get_parameter_variance()
        self._var_current = self.gp_calculator.get_parameter_variance()
        self._fitted = True

    def _ensure_fresh_data(self) -> None:
        """确保每次 forward 使用最新 Dt。"""
        if not hasattr(self.model, "train_inputs") or self.model.train_inputs is None:
            return
        X_t = self.model.train_inputs[0]
        y_t = self.model.train_targets
        if X_t is None or y_t is None:
            return
        n = X_t.shape[0]
        if (not self._fitted) or (n != self._last_hist_n):
            X_np = X_t.detach().cpu().numpy()
            y_np = y_t.detach().cpu().numpy()
            self._fit_internal(X_np, y_np)
            self._last_hist_n = n
            # 首次可用时推断 variable_types
            self._maybe_infer_variable_types()

    def _compute_dynamic_lambda(self) -> float:
        if not self._fitted or self._var_current is None or self._var_initial is None:
            return self.lambda_min
        offset = 1 if self.gp_calculator.include_intercept else 0
        n_main = self._n_features or 0
        var_cur = self._var_current[offset : offset + n_main]
        var_ini = self._var_initial[offset : offset + n_main]
        valid = var_ini > 1e-10
        if not np.any(valid):
            r_t = 1.0
        else:
            r_t = np.mean(var_cur[valid] / var_ini[valid])
        if r_t > self.tau_1:
            return self.lambda_min
        if r_t < self.tau_2:
            return self.lambda_max
        # 线性插值
        return self.lambda_min + (self.lambda_max - self.lambda_min) * (
            (self.tau_1 - r_t) / (self.tau_1 - self.tau_2)
        )

    def _compute_info_gain(self, X_cand: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call after data refresh"
        n = X_cand.shape[0]
        scores = np.zeros(n)
        lam = self._compute_dynamic_lambda()
        # 调试：跟踪 ΔVar 的全局最小/最大
        vr_main_min = np.inf
        vr_main_max = -np.inf
        vr_inter_min = np.inf
        vr_inter_max = -np.inf
        for i in range(n):
            x = X_cand[i : i + 1]
            main_red, inter_red = self.gp_calculator.compute_variance_reduction(x)
            # 数值稳定：将极小负值截断为 0
            if main_red is not None and len(main_red) > 0:
                main_red = np.maximum(main_red, 0.0)
                vr_main_min = min(vr_main_min, float(np.min(main_red)))
                vr_main_max = max(vr_main_max, float(np.max(main_red)))
            if inter_red is not None and len(inter_red) > 0:
                inter_red = np.maximum(inter_red, 0.0)
                vr_inter_min = min(vr_inter_min, float(np.min(inter_red)))
                vr_inter_max = max(vr_inter_max, float(np.max(inter_red)))
            avg_main = float(np.mean(main_red)) if len(main_red) > 0 else 0.0
            avg_inter = float(np.mean(inter_red)) if len(inter_red) > 0 else 0.0
            scores[i] = avg_main + lam * avg_inter
        # 保存最新 ΔVar 统计，供 forward 中调试输出
        if self.debug_components:
            # 若所有候选都为空交互/主效应，则保持为 None
            self._last_vr_main_min = None if vr_main_min is np.inf else vr_main_min
            self._last_vr_main_max = None if vr_main_max == -np.inf else vr_main_max
            self._last_vr_inter_min = None if vr_inter_min is np.inf else vr_inter_min
            self._last_vr_inter_max = None if vr_inter_max == -np.inf else vr_inter_max
        return scores

    def _compute_coverage(self, X_cand: np.ndarray) -> np.ndarray:
        assert self._fitted and self._X_train is not None
        # 将 'integer' 归一为覆盖度中的 'continuous'
        vt = None
        if self.variable_types is not None:
            vt = {
                k: ("categorical" if v == "categorical" else "continuous")
                for k, v in self.variable_types.items()
            }

        cov = compute_coverage_batch(
            X_cand,
            self._X_train,
            variable_types=vt,
            ranges=None,
            method=self.coverage_method,
        )
        return self.gamma * cov

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 1) 确保使用最新 Dt
        self._ensure_fresh_data()
        if not self._fitted:
            return torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

        # 2) 处理批/q 维，转 numpy
        X_np = X.detach().cpu().numpy()
        if X_np.ndim == 3:
            B, q, d = X_np.shape
            flat = X_np.reshape(-1, d)
        else:
            B, q, d = X_np.shape[0], 1, X_np.shape[1]
            flat = X_np

        # 3) 规范化到与历史一致的变换空间
        flat_can = self._canonicalize_numpy(flat)

        # 4) 计算 EUR 组成
        info = self._compute_info_gain(flat_can)
        cov = self._compute_coverage(flat_can)
        total = info + cov

        # 可选：记录并打印组件统计（仅用于调试）
        if self.debug_components:
            self._last_info = info.copy()
            self._last_cov = cov.copy()
            try:
                info_min, info_max, info_mean = (
                    float(np.min(info)),
                    float(np.max(info)),
                    float(np.mean(info)),
                )
                cov_min, cov_max, cov_mean = (
                    float(np.min(cov)),
                    float(np.max(cov)),
                    float(np.mean(cov)),
                )
                lam = self._compute_dynamic_lambda()
                msg = (
                    f"[EURAcqfV4] components: info[min/mean/max]={info_min:.4f}/{info_mean:.4f}/{info_max:.4f}, "
                    f"cov(γ={self.gamma})[min/mean/max]={cov_min:.4f}/{cov_mean:.4f}/{cov_max:.4f}, "
                    f"λ_t={lam:.3f}"
                )
                # 若存在 ΔVar 统计，则追加打印
                if hasattr(self, "_last_vr_main_min"):
                    vr_main_min = getattr(self, "_last_vr_main_min", None)
                    vr_main_max = getattr(self, "_last_vr_main_max", None)
                    vr_inter_min = getattr(self, "_last_vr_inter_min", None)
                    vr_inter_max = getattr(self, "_last_vr_inter_max", None)
                    extra = []
                    if vr_main_min is not None and vr_main_max is not None:
                        extra.append(
                            f"ΔVar_main[min/max]={vr_main_min:.4e}/{vr_main_max:.4e}"
                        )
                    if vr_inter_min is not None and vr_inter_max is not None:
                        extra.append(
                            f"ΔVar_inter[min/max]={vr_inter_min:.4e}/{vr_inter_max:.4e}"
                        )
                    if len(extra) > 0:
                        msg += "; " + ", ".join(extra)
                print(msg)
            except Exception:
                pass

        # 5) 还原到 batch，并在 q 上聚合
        total_t = torch.from_numpy(total).to(dtype=X.dtype, device=X.device)
        if q > 1:
            total_t = total_t.view(B, q).sum(dim=-1)
        else:
            total_t = total_t.view(B)
        return total_t

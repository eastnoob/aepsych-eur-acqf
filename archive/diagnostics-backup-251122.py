"""
诊断工具模块

提供完整的调试和诊断功能：
- 动态权重状态
- 效应贡献分析
- 模型状态检查
- 配置参数验证

Example:
    >>> diagnostics = DiagnosticsManager(debug_components=True)
    >>> diagnostics.update_effects(main_sum, pair_sum, triplet_sum, info_raw, cov)
    >>> diagnostics.print_diagnostics()
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from loguru import logger


class DiagnosticsManager:
    """诊断信息管理器

    收集和展示采集函数的运行时状态。
    """

    def __init__(self, debug_components: bool = False):
        """
        Args:
            debug_components: 是否启用分量缓存（会轻微影响性能）
        """
        self.debug_components = debug_components

        # 效应缓存（调试用）
        self._last_main: Optional[torch.Tensor] = None
        self._last_pair: Optional[torch.Tensor] = None
        self._last_triplet: Optional[torch.Tensor] = None
        self._last_info: Optional[torch.Tensor] = None
        self._last_cov: Optional[torch.Tensor] = None

    def update_effects(
        self,
        main_sum: Optional[torch.Tensor] = None,
        pair_sum: Optional[torch.Tensor] = None,
        triplet_sum: Optional[torch.Tensor] = None,
        info_raw: Optional[torch.Tensor] = None,
        cov: Optional[torch.Tensor] = None,
    ) -> None:
        """更新效应贡献缓存

        Args:
            main_sum: 主效应总和
            pair_sum: 二阶交互总和
            triplet_sum: 三阶交互总和
            info_raw: 信息项（未标准化）
            cov: 覆盖项
        """
        if self.debug_components:
            if main_sum is not None:
                self._last_main = main_sum.detach().cpu()
            if pair_sum is not None:
                self._last_pair = pair_sum.detach().cpu()
            if triplet_sum is not None:
                self._last_triplet = triplet_sum.detach().cpu()
            if info_raw is not None:
                self._last_info = info_raw.detach().cpu()
            if cov is not None:
                self._last_cov = cov.detach().cpu()

    def get_diagnostics(
        self,
        lambda_t: float,
        gamma_t: float,
        lambda_2: Optional[float] = None,
        lambda_3: Optional[float] = None,
        n_train: int = 0,
        fitted: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """获取诊断信息字典

        Args:
            lambda_t: 二阶交互权重
            gamma_t: 覆盖权重
            lambda_2: 二阶权重（显式）
            lambda_3: 三阶权重
            n_train: 训练样本数
            fitted: 是否已拟合
            config: 配置参数字典

        Returns:
            完整的诊断信息字典
        """
        diag = {
            # 动态权重
            "lambda_t": lambda_t,
            "lambda_2": lambda_2 if lambda_2 is not None else lambda_t,
            "lambda_3": lambda_3 if lambda_3 is not None else 0.0,
            "gamma_t": gamma_t,
            # 模型状态
            "n_train": n_train,
            "fitted": fitted,
        }

        # 配置参数
        if config is not None:
            diag.update(config)

        # 效应贡献（如果启用调试）
        if self.debug_components:
            if self._last_main is not None:
                diag["main_effects_sum"] = self._last_main
            if self._last_pair is not None:
                diag["pair_effects_sum"] = self._last_pair
            if self._last_triplet is not None:
                diag["triplet_effects_sum"] = self._last_triplet
            if self._last_info is not None:
                diag["info_raw"] = self._last_info
            if self._last_cov is not None:
                diag["coverage"] = self._last_cov

        return diag

    def print_diagnostics(self, diag: Dict[str, Any], verbose: bool = False) -> None:
        """打印诊断信息到控制台

        Args:
            diag: 诊断字典（从get_diagnostics获取）
            verbose: 是否打印详细信息（包括完整数组）
        """
        logger.info("\n" + "=" * 70)
        logger.info("EURAnovaPairAcqf 诊断信息")
        logger.info("=" * 70)

        logger.info(f"\n【动态权重状态】")
        logger.info(f"  λ_2 (二阶交互权重) = {diag.get('lambda_2', 0):.4f}")

        if diag.get("lambda_3", 0) > 0:
            logger.info(f"  λ_3 (三阶交互权重) = {diag.get('lambda_3', 0):.4f}")

        logger.info(f"  γ_t (覆盖权重) = {diag.get('gamma_t', 0):.4f}")

        if "lambda_min" in diag:
            logger.info(
                f"  λ 范围: [{diag.get('lambda_min', 0):.2f}, {diag.get('lambda_max', 1):.2f}]"
            )

        if "gamma_min" in diag:
            logger.info(
                f"  γ 范围: [{diag.get('gamma_min', 0):.2f}, {diag.get('gamma_max', 1):.2f}]"
            )

        logger.info(f"\n【模型状态】")
        logger.info(f"  训练样本数: {diag.get('n_train', 0)}")

        if "tau_n_min" in diag and "tau_n_max" in diag:
            logger.info(
                f"  转向阈值: tau_n_min={diag['tau_n_min']}, "
                f"tau_n_max={diag['tau_n_max']}"
            )

        logger.info(f"  模型已拟合: {'是' if diag.get('fitted', False) else '否'}")

        logger.info(f"\n【交互配置】")

        if "n_pairs" in diag:
            logger.info(f"  二阶交互数量: {diag['n_pairs']}")
            if diag["n_pairs"] > 0 and "pairs" in diag:
                pairs_str = ", ".join([f"({i},{j})" for i, j in diag["pairs"][:5]])
                if diag["n_pairs"] > 5:
                    pairs_str += f", ... (共{diag['n_pairs']}个)"
                logger.info(f"  二阶交互: {pairs_str}")

        if "n_triplets" in diag:
            logger.info(f"  三阶交互数量: {diag['n_triplets']}")
            if diag["n_triplets"] > 0 and "triplets" in diag:
                triplets_str = ", ".join(
                    [f"({i},{j},{k})" for i, j, k in diag["triplets"][:3]]
                )
                if diag["n_triplets"] > 3:
                    triplets_str += f", ... (共{diag['n_triplets']}个)"
                logger.info(f"  三阶交互: {triplets_str}")

        # 效应贡献
        if "main_effects_sum" in diag:
            logger.info(f"\n【效应贡献】(最后一次 forward() 调用)")

            main = diag["main_effects_sum"]
            logger.info(f"  主效应总和: mean={main.mean():.4f}, std={main.std():.4f}")

            if "pair_effects_sum" in diag:
                pair = diag["pair_effects_sum"]
                logger.info(
                    f"  二阶交互总和: mean={pair.mean():.4f}, std={pair.std():.4f}"
                )

            if "triplet_effects_sum" in diag:
                triplet = diag["triplet_effects_sum"]
                logger.info(
                    f"  三阶交互总和: mean={triplet.mean():.4f}, std={triplet.std():.4f}"
                )

            if "info_raw" in diag:
                info = diag["info_raw"]
                logger.info(f"  信息项: mean={info.mean():.4f}, std={info.std():.4f}")

            if "coverage" in diag:
                cov = diag["coverage"]
                logger.info(f"  覆盖项: mean={cov.mean():.4f}, std={cov.std():.4f}")

            if verbose:
                logger.debug(f"\n  主效应数组:\n    {main}")
                if "pair_effects_sum" in diag:
                    logger.debug(f"  二阶交互数组:\n    {pair}")
                if "triplet_effects_sum" in diag:
                    logger.debug(f"  三阶交互数组:\n    {triplet}")

        else:
            logger.warning(
                f"\n⚠️  效应贡献数据不可用 - 提示: 初始化时设置 debug_components=True"
            )

        logger.info("=" * 70 + "\n")

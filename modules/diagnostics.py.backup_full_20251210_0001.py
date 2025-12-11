"""
Full backup snapshot of `diagnostics.py` created 2025-12-10 00:01 (auto)
"""

```python
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
from pathlib import Path
from datetime import datetime


class DiagnosticsManager:
    """诊断信息管理器

    收集和展示采集函数的运行时状态。

    输出行为受三个选项控制：
      - enabled: 是否在运行时打印精简诊断（每次采样/forward 可见）
      - verbose_mode: 是否打印完整数组/详细内容（应为 debug 级别或写文件）
      - output_file: 可选路径，将 verbose 模式下的完整诊断写入文件以便离线分析
    """

    def __init__(
        self,
        debug_components: bool = False,
        enabled: bool = False,
        verbose_mode: bool = False,
        output_file: Optional[str] = None,
    ):
        """
        Args:
            debug_components: 是否启用分量缓存（会微量影响性能）
            enabled: 是否在运行时打印精简诊断（INFO 级别）
            verbose_mode: 是否打印完整诊断（DEBUG 级别或写入文件）
            output_file: 若指定，verbose 输出将写入此文件
        """
        self.debug_components = debug_components
        self.enabled = enabled
        self.verbose_mode = verbose_mode
        self.output_file = Path(output_file) if output_file else None

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
        cov: Optional[torch.Tensor] = None
    ) -> None:
        """更新效应贡献缓存

        Args:
            main_sum: 主效应总和
            pair_sum: 二阶交互总和
            triplet_sum: 三阶交互总和
            info_raw: 信息项（未标准化）
            cov: 覆盖项
        """
        # Determine template to safely coerce shapes
        template = None
        for t in (main_sum, pair_sum, triplet_sum, info_raw, cov):
            if t is not None:
                try:
                    template = t.detach().cpu()
                    break
                except Exception:
                    template = None

        if template is None:
            template = torch.zeros(1, dtype=torch.float32)

        def _safe_to_cpu(x: Optional[torch.Tensor]) -> torch.Tensor:
            if x is None:
                try:
                    return torch.zeros_like(template)
                except Exception:
                    return torch.zeros(1, dtype=torch.float32)
            try:
                return x.detach().cpu()
            except Exception:
                t = torch.as_tensor(x)
                if t.numel() == 1 and template.numel() > 1:
                    return torch.zeros_like(template)
                return t.detach().cpu()

        self._last_main = _safe_to_cpu(main_sum)
        self._last_pair = _safe_to_cpu(pair_sum)
        self._last_triplet = _safe_to_cpu(triplet_sum)
        self._last_info = _safe_to_cpu(info_raw)
        self._last_cov = _safe_to_cpu(cov)

    def get_diagnostics(
        self,
        lambda_t: float,
        gamma_t: float,
        lambda_2: Optional[float] = None,
        lambda_3: Optional[float] = None,
        n_train: int = 0,
        fitted: bool = False,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        diag = {
            "lambda_t": lambda_t,
            "lambda_2": lambda_2 if lambda_2 is not None else lambda_t,
            "lambda_3": lambda_3 if lambda_3 is not None else 0.0,
            "gamma_t": gamma_t,
            "n_train": n_train,
            "fitted": fitted,
        }
        if config is not None:
            diag.update(config)
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
        """控制化记录诊断信息。

        - enabled=False: only warnings/errors
        - enabled=True: emits concise INFO summary
        - verbose or verbose_mode: emits DEBUG arrays and optionally writes file
        """
        if not self.enabled and not verbose and not self.verbose_mode:
            return

        try:
            lambda2 = diag.get('lambda_2', diag.get('lambda_t', 0.0))
            gamma_t = diag.get('gamma_t', 0.0)
            n_train = diag.get('n_train', 0)
            fitted = diag.get('fitted', False)
            summary = (
                f"[EUR] n_train={n_train} fitted={int(bool(fitted))} "
                f"λ₂={lambda2:.3f} γ={gamma_t:.3f}"
            )
        except Exception:
            summary = "[EUR] diagnostics summary unavailable"

        if self.enabled:
            logger.info(summary)

        if verbose or self.verbose_mode:
            logger.debug("--- EUR Detailed Diagnostics BEGIN ---")
            logger.debug(f"lambda_2={diag.get('lambda_2', None)} lambda_3={diag.get('lambda_3', None)}")
            logger.debug(f"gamma_t={diag.get('gamma_t', None)} n_train={diag.get('n_train', None)}")

            if 'main_effects_sum' in diag:
                main = diag['main_effects_sum']
                logger.debug(f"main: mean={main.mean():.6f} std={main.std():.6f} shape={tuple(main.shape)}")
            if 'pair_effects_sum' in diag:
                pair = diag['pair_effects_sum']
                logger.debug(f"pair: mean={pair.mean():.6f} std={pair.std():.6f} shape={tuple(pair.shape)}")
            if 'triplet_effects_sum' in diag:
                triplet = diag['triplet_effects_sum']
                logger.debug(f"triplet: mean={triplet.mean():.6f} std={triplet.std():.6f} shape={tuple(triplet.shape)}")

            if self.output_file is not None:
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    outpath = self.output_file.with_name(self.output_file.stem + f"_{ts}" + self.output_file.suffix)
                    with open(outpath, "w", encoding="utf-8") as fh:
                        fh.write("EUR Detailed Diagnostics\n")
                        for k, v in diag.items():
                            fh.write(f"{k}: {repr(v)}\n")
                    logger.info(f"Wrote detailed diagnostics to {outpath}")
                except Exception as e:
                    logger.error(f"Failed to write diagnostics to {self.output_file}: {e}")

            logger.debug("--- EUR Detailed Diagnostics END ---")

        if "main_effects_sum" in diag:
            main = diag["main_effects_sum"]
            logger.info("\n【效应贡献】(最后一次 forward() 调用)")
            logger.info(f"  主效应总和: mean={main.mean():.4f}, std={main.std():.4f}")

            if "pair_effects_sum" in diag:
                pair = diag["pair_effects_sum"]
                logger.info(f"  二阶交互总和: mean={pair.mean():.4f}, std={pair.std():.4f}")

            if "triplet_effects_sum" in diag:
                triplet = diag["triplet_effects_sum"]
                logger.info(f"  三阶交互总和: mean={triplet.mean():.4f}, std={triplet.std():.4f}")

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
            logger.warning("\n⚠️  效应贡献数据不可用 - 提示: 初始化时设置 debug_components=True")

        logger.info("=" * 70 + "\n")

```

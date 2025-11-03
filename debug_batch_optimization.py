"""
简化调试脚本：对比单个候选点的计算过程
"""

import torch
import numpy as np
import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "temp_aepsych"))

from extensions.dynamic_eur_acquisition.eur_anova_pair_acquisition import (
    EURAnovaPairAcqf,
)
from extensions.dynamic_eur_acquisition.eur_anova_pair_acquisition_optimized import (
    EURAnovaPairAcqf_BatchOptimized,
)
from aepsych.models import OrdinalGPModel
from aepsych.likelihoods import OrdinalLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel


def create_test_model():
    """创建简单测试模型"""
    torch.manual_seed(42)
    X_train = torch.rand(20, 6)
    y_train = torch.randint(0, 5, (20,)).float()

    likelihood = OrdinalLikelihood(n_levels=5)
    model = OrdinalGPModel(
        dim=6,
        likelihood=likelihood,
        covar_module=ScaleKernel(RBFKernel(ard_num_dims=6)),
    )
    model.fit(X_train, y_train)
    return model


def debug_single_point():
    """调试单个候选点的计算"""
    print("=" * 80)
    print("单点调试：对比原始 vs 优化版本")
    print("=" * 80)

    model = create_test_model()

    config = {
        "gamma": 0.25,
        "main_weight": 1.0,
        "use_dynamic_lambda": True,
        "tau1": 0.7,
        "tau2": 0.3,
        "lambda_min": 0.1,
        "lambda_max": 1.0,
        "interaction_pairs": "0,1;2,3;4,5",
        "local_jitter_frac": 0.08,
        "local_num": 4,
        "variable_types_list": "categorical, integer, integer, continuous, categorical, integer",
        "debug_components": True,  # 启用调试模式
    }

    acqf_original = EURAnovaPairAcqf(model, **config)
    acqf_optimized = EURAnovaPairAcqf_BatchOptimized(model, **config)

    # 测试点
    torch.manual_seed(123)
    X_test = torch.rand(1, 6)

    print(f"\n测试候选点: {X_test.numpy()}")
    print(f"维度: {X_test.shape[1]}")
    print(f"交互对数: {len(acqf_original._pairs)}")
    print(f"local_num: {config['local_num']}")

    # 计算结果
    with torch.no_grad():
        result_original = acqf_original(X_test)
        result_optimized = acqf_optimized(X_test)

    print(f"\n【原始版本】")
    print(f"  最终结果: {result_original.item():.8f}")
    if hasattr(acqf_original, "_last_main"):
        print(f"  主效应项: {acqf_original._last_main.numpy()}")
    if hasattr(acqf_original, "_last_pair"):
        print(f"  交互效应项: {acqf_original._last_pair.numpy()}")
    if hasattr(acqf_original, "_last_info"):
        print(f"  信息项: {acqf_original._last_info.item():.8f}")
    if hasattr(acqf_original, "_last_cov"):
        print(f"  覆盖项: {acqf_original._last_cov.item():.8f}")

    print(f"\n【优化版本】")
    print(f"  最终结果: {result_optimized.item():.8f}")
    if hasattr(acqf_optimized, "_last_main"):
        print(f"  主效应项: {acqf_optimized._last_main.numpy()}")
    if hasattr(acqf_optimized, "_last_pair"):
        print(f"  交互效应项: {acqf_optimized._last_pair.numpy()}")
    if hasattr(acqf_optimized, "_last_info"):
        print(f"  信息项: {acqf_optimized._last_info.item():.8f}")
    if hasattr(acqf_optimized, "_last_cov"):
        print(f"  覆盖项: {acqf_optimized._last_cov.item():.8f}")

    print(f"\n【差异】")
    abs_diff = abs(result_original.item() - result_optimized.item())
    rel_diff = abs_diff / (abs(result_original.item()) + 1e-8)
    print(f"  绝对差异: {abs_diff:.8f}")
    print(f"  相对差异: {rel_diff:.2%}")

    if abs_diff < 1e-6:
        print("\n✅ 数值一致！")
        return True
    else:
        print("\n❌ 数值不一致！")
        return False


if __name__ == "__main__":
    debug_single_point()

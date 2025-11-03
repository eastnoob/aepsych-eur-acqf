"""
测试不同规模设计下的性能提升

目标：验证维度和交互对数量增加时，批量优化的加速比是否提升
"""

import torch
import numpy as np
import time
from typing import Dict, Any
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


def create_model(dim, n_train=50):
    """创建指定维度的模型"""
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.rand(n_train, dim)
    y_train = torch.randint(0, 5, (n_train,)).float()

    likelihood = OrdinalLikelihood(n_levels=5)
    model = OrdinalGPModel(
        dim=dim,
        likelihood=likelihood,
        covar_module=ScaleKernel(RBFKernel(ard_num_dims=dim)),
    )
    model.fit(X_train, y_train)
    return model


def count_metric_calls(acqf, X_test):
    """统计 _metric 调用次数"""
    call_count = 0
    original_metric = acqf._metric

    def wrapped_metric(X):
        nonlocal call_count
        call_count += 1
        return original_metric(X)

    acqf._metric = wrapped_metric
    with torch.no_grad():
        _ = acqf(X_test)
    acqf._metric = original_metric

    return call_count


def test_scaling(dim, n_pairs, n_candidates=30):
    """测试指定维度和交互对数量的性能"""
    print(f"\n{'='*80}")
    print(f"测试配置: dim={dim}, 交互对数={n_pairs}, 候选点数={n_candidates}")
    print(f"{'='*80}")

    # 创建模型
    model = create_model(dim)

    # 生成交互对（相邻配对）
    pairs = [(i, i + 1) for i in range(0, min(dim - 1, n_pairs * 2), 2)][:n_pairs]
    pairs_str = ";".join([f"{i},{j}" for i, j in pairs])

    # 生成变量类型（交替 categorical, integer, continuous）
    var_types = []
    for i in range(dim):
        if i % 3 == 0:
            var_types.append("categorical")
        elif i % 3 == 1:
            var_types.append("integer")
        else:
            var_types.append("continuous")
    var_types_str = ", ".join(var_types)

    config = {
        "gamma": 0.25,
        "main_weight": 1.0,
        "use_dynamic_lambda": True,
        "tau1": 0.7,
        "tau2": 0.3,
        "lambda_min": 0.1,
        "lambda_max": 1.0,
        "interaction_pairs": pairs_str,
        "local_jitter_frac": 0.08,
        "local_num": 4,
        "variable_types_list": var_types_str,
    }

    print(f"交互对: {pairs}")
    print(
        f"变量类型: {var_types[:6]}..."
        if len(var_types) > 6
        else f"变量类型: {var_types}"
    )

    # 创建采集函数
    try:
        acqf_original = EURAnovaPairAcqf(model, **config)
        acqf_optimized = EURAnovaPairAcqf_BatchOptimized(model, **config)
    except Exception as e:
        print(f"❌ 创建采集函数失败: {e}")
        return None

    # 准备测试数据
    torch.manual_seed(456)
    X_test = torch.rand(n_candidates, dim)

    # 理论模型调用次数
    theoretical_calls = dim + n_pairs  # 主效应 + 交互效应
    print(f"理论模型调用次数 (原始): {dim} + {n_pairs} = {theoretical_calls} 次/候选点")

    # 测试原始版本
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_candidates):
            _ = acqf_original(X_test[i : i + 1])
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_original = time.time() - t0

    calls_original = count_metric_calls(acqf_original, X_test[0:1])

    # 测试优化版本
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_candidates):
            _ = acqf_optimized(X_test[i : i + 1])
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_optimized = time.time() - t0

    calls_optimized = count_metric_calls(acqf_optimized, X_test[0:1])

    # 计算性能提升
    speedup = time_original / time_optimized if time_optimized > 0 else float("inf")
    call_reduction = (
        calls_original / calls_optimized if calls_optimized > 0 else float("inf")
    )

    print(f"\n【原始版本】")
    print(f"  耗时: {time_original:.3f}秒")
    print(f"  _metric 调用: {calls_original} 次/候选点")

    print(f"\n【优化版本】")
    print(f"  耗时: {time_optimized:.3f}秒")
    print(f"  _metric 调用: {calls_optimized} 次/候选点")

    print(f"\n【性能提升】")
    print(f"  ⚡ 加速比: {speedup:.2f}x")
    print(f"  📉 模型调用减少: {call_reduction:.2f}x")
    print(
        f"  💾 时间节省: {time_original - time_optimized:.3f}秒 ({(1-time_optimized/time_original)*100:.1f}%)"
    )

    return {
        "dim": dim,
        "n_pairs": n_pairs,
        "theoretical_calls": theoretical_calls,
        "calls_original": calls_original,
        "calls_optimized": calls_optimized,
        "time_original": time_original,
        "time_optimized": time_optimized,
        "speedup": speedup,
        "call_reduction": call_reduction,
    }


def main():
    """测试不同规模的设计"""
    print("=" * 80)
    print("批量优化性能 - 规模缩放测试")
    print("=" * 80)

    # 测试配置：(维度, 交互对数, 候选点数)
    test_configs = [
        (6, 3, 30),  # 小规模（当前linear实验）
        (10, 5, 30),  # 中等规模
        (15, 7, 30),  # 较大规模
        (20, 10, 20),  # 大规模（减少候选点以加快测试）
    ]

    results = []
    for dim, n_pairs, n_candidates in test_configs:
        result = test_scaling(dim, n_pairs, n_candidates)
        if result:
            results.append(result)

    # 汇总结果
    print(f"\n{'='*80}")
    print("汇总结果")
    print(f"{'='*80}")
    print(
        f"\n{'维度':<6} {'交互对':<8} {'理论调用':<10} {'实际调用(原)':<14} {'实际调用(优)':<14} {'加速比':<10} {'调用减少':<10}"
    )
    print(f"{'-'*80}")

    for r in results:
        print(
            f"{r['dim']:<6} {r['n_pairs']:<8} {r['theoretical_calls']:<10} "
            f"{r['calls_original']:<14} {r['calls_optimized']:<14} "
            f"{r['speedup']:<10.2f} {r['call_reduction']:<10.2f}"
        )

    # 分析趋势
    print(f"\n{'='*80}")
    print("趋势分析")
    print(f"{'='*80}")

    if len(results) >= 2:
        speedups = [r["speedup"] for r in results]
        dims = [r["dim"] for r in results]

        print(f"\n维度增加时的加速比变化:")
        for i, (d, s) in enumerate(zip(dims, speedups)):
            print(f"  dim={d:2d}: {s:.2f}x")

        if speedups[-1] > speedups[0]:
            improvement = (speedups[-1] / speedups[0] - 1) * 100
            print(f"\n✅ 加速比随维度增加而提升")
            print(
                f"   从 {speedups[0]:.2f}x 提升到 {speedups[-1]:.2f}x (提升 {improvement:.1f}%)"
            )
        else:
            print(f"\n⚠️ 加速比未随维度明显提升")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

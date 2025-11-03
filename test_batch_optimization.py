"""
性能对比测试：EURAnovaPairAcqf vs EURAnovaPairAcqf_BatchOptimized

测试目标：
1. 验证数值一致性（优化前后结果应完全相同）
2. 测量性能提升（墙钟时间、模型调用次数）
3. 检查内存占用（批量计算可能消耗更多内存）
"""

import torch
import numpy as np
import time
from typing import Dict, Any
import sys
import os

# 添加路径
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "temp_aepsych"))

# 导入采集函数
from extensions.dynamic_eur_acquisition.eur_anova_pair_acquisition import (
    EURAnovaPairAcqf,
)
from extensions.dynamic_eur_acquisition.eur_anova_pair_acquisition_optimized import (
    EURAnovaPairAcqf_BatchOptimized,
)

# 导入模型相关
from aepsych.models import OrdinalGPModel
from aepsych.likelihoods import OrdinalLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel


def create_test_model(n_train=30, dim=6, n_levels=5):
    """创建测试模型"""
    # 生成训练数据
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.rand(n_train, dim)
    y_train = torch.randint(0, n_levels, (n_train,)).float()

    # 创建模型
    likelihood = OrdinalLikelihood(n_levels=n_levels)
    model = OrdinalGPModel(
        dim=dim,
        likelihood=likelihood,
        covar_module=ScaleKernel(RBFKernel(ard_num_dims=dim)),
    )

    # 拟合数据
    model.fit(X_train, y_train)

    return model, X_train, y_train


def count_metric_calls(acqf, X_test):
    """统计 _metric 调用次数"""
    call_count = 0
    original_metric = acqf._metric

    def wrapped_metric(X):
        nonlocal call_count
        call_count += 1
        return original_metric(X)

    acqf._metric = wrapped_metric
    _ = acqf(X_test)
    acqf._metric = original_metric

    return call_count


def test_numerical_consistency():
    """测试1：验证数值一致性"""
    print("\n" + "=" * 80)
    print("测试1: 数值一致性验证")
    print("=" * 80)

    # 创建模型
    model, X_train, y_train = create_test_model()

    # 配置参数（与linear实验相同）
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
    }

    # 创建两个采集函数
    acqf_original = EURAnovaPairAcqf(model, **config)
    acqf_optimized = EURAnovaPairAcqf_BatchOptimized(model, **config)

    # 准备测试候选点
    torch.manual_seed(123)
    X_test = torch.rand(10, 6)  # 10个候选点

    # 计算结果 (逐个评估，因为采集函数要求 q=1)
    result_original = []
    result_optimized = []
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            # 为每个候选点设置相同的随机种子
            torch.manual_seed(1000 + i)
            np.random.seed(1000 + i)
            result_original.append(acqf_original(X_test[i : i + 1]))

            torch.manual_seed(1000 + i)
            np.random.seed(1000 + i)
            result_optimized.append(acqf_optimized(X_test[i : i + 1]))

    result_original = torch.cat(result_original)
    result_optimized = torch.cat(result_optimized)

    # 数值比较
    abs_diff = torch.abs(result_original - result_optimized)
    rel_diff = abs_diff / (torch.abs(result_original) + 1e-8)

    print(f"\n结果形状: {result_original.shape}")
    print(f"原始版本: {result_original[:5].numpy()}")
    print(f"优化版本: {result_optimized[:5].numpy()}")
    print(
        f"\n绝对差异: max={abs_diff.max().item():.2e}, mean={abs_diff.mean().item():.2e}"
    )
    print(
        f"相对差异: max={rel_diff.max().item():.2e}, mean={rel_diff.mean().item():.2e}"
    )

    # 判断是否一致
    is_close = torch.allclose(result_original, result_optimized, atol=1e-6, rtol=1e-5)

    if is_close:
        print("\n✅ 数值一致性验证通过！")
        return True
    else:
        print("\n❌ 数值一致性验证失败！")
        print(f"   差异最大的索引: {abs_diff.argmax().item()}")
        print(f"   原始值: {result_original[abs_diff.argmax()]}")
        print(f"   优化值: {result_optimized[abs_diff.argmax()]}")
        return False


def test_performance():
    """测试2：性能对比"""
    print("\n" + "=" * 80)
    print("测试2: 性能对比")
    print("=" * 80)

    # 创建模型
    model, X_train, y_train = create_test_model(n_train=50)  # 更多训练数据

    # 配置参数
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
    }

    # 创建两个采集函数
    acqf_original = EURAnovaPairAcqf(model, **config)
    acqf_optimized = EURAnovaPairAcqf_BatchOptimized(model, **config)

    # 准备测试候选点（模拟实际优化场景）
    torch.manual_seed(456)
    n_candidates = 50  # 模拟BoTorch优化的候选点数量（减少以加快测试）
    X_test = torch.rand(n_candidates, 6)

    # 测试原始版本
    print(f"\n测试配置:")
    print(f"  - 候选点数: {n_candidates}")
    print(f"  - 维度: 6")
    print(f"  - 交互对数: {len(acqf_original._pairs)}")
    print(f"  - local_num: {config['local_num']}")
    print(
        f"  - 预期模型调用 (原始): {6 + len(acqf_original._pairs)} × {config['local_num']} = {(6 + len(acqf_original._pairs)) * config['local_num']} 次/候选点"
    )

    print("\n【原始版本】")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_candidates):
            _ = acqf_original(X_test[i : i + 1])
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_original = time.time() - t0

    calls_original = count_metric_calls(
        acqf_original, X_test[0:1]
    )  # 单个候选点的调用次数

    print(f"  耗时: {time_original:.3f}秒 ({n_candidates}个候选点)")
    print(f"  _metric 调用次数 (单个候选点): {calls_original}")

    # 测试优化版本
    print("\n【优化版本】")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_candidates):
            _ = acqf_optimized(X_test[i : i + 1])
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_optimized = time.time() - t0

    calls_optimized = count_metric_calls(acqf_optimized, X_test[0:1])

    print(f"  耗时: {time_optimized:.3f}秒 ({n_candidates}个候选点)")
    print(f"  _metric 调用次数 (单个候选点): {calls_optimized}")

    # 性能提升
    speedup = time_original / time_optimized
    call_reduction = (
        calls_original / calls_optimized if calls_optimized > 0 else float("inf")
    )

    print(f"\n【性能提升】")
    print(f"  ⚡ 加速比: {speedup:.2f}x")
    print(f"  📉 模型调用减少: {call_reduction:.2f}x")
    print(
        f"  💾 时间节省: {time_original - time_optimized:.3f}秒 ({(1-time_optimized/time_original)*100:.1f}%)"
    )

    if speedup > 5:
        print("\n✅ 性能提升显著（>5x）")
    elif speedup > 2:
        print("\n✅ 性能提升明显（>2x）")
    else:
        print("\n⚠️ 性能提升有限（<2x）")


def test_memory_usage():
    """测试3：内存占用对比"""
    print("\n" + "=" * 80)
    print("测试3: 内存占用对比")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ℹ️ GPU不可用，跳过GPU内存测试")
        return

    # 创建GPU模型
    model, _, _ = create_test_model(n_train=50)
    model = model.cuda()

    config = {
        "gamma": 0.25,
        "main_weight": 1.0,
        "interaction_pairs": "0,1;2,3;4,5",
        "local_num": 4,
        "variable_types_list": "categorical, integer, integer, continuous, categorical, integer",
    }

    acqf_original = EURAnovaPairAcqf(model, **config)
    acqf_optimized = EURAnovaPairAcqf_BatchOptimized(model, **config)

    X_test = torch.rand(20, 6).cuda()

    # 测试原始版本
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            _ = acqf_original(X_test[i : i + 1])
    mem_original = torch.cuda.max_memory_allocated() / 1024**2  # MB

    # 测试优化版本
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            _ = acqf_optimized(X_test[i : i + 1])
    mem_optimized = torch.cuda.max_memory_allocated() / 1024**2  # MB

    print(f"\nGPU 峰值内存占用:")
    print(f"  原始版本: {mem_original:.2f} MB")
    print(f"  优化版本: {mem_optimized:.2f} MB")
    print(
        f"  增加: {mem_optimized - mem_original:.2f} MB ({(mem_optimized/mem_original - 1)*100:.1f}%)"
    )

    if mem_optimized < mem_original * 2:
        print("\n✅ 内存增长可接受（<2x）")
    else:
        print("\n⚠️ 内存增长较大（>2x）")


def main():
    """运行所有测试"""
    print("=" * 80)
    print("EURAnovaPairAcqf 批量性能优化 - 对比测试")
    print("=" * 80)

    # 测试1: 数值一致性
    consistency_passed = test_numerical_consistency()

    if not consistency_passed:
        print("\n❌ 数值一致性验证失败，停止后续测试")
        return

    # 测试2: 性能对比
    test_performance()

    # 测试3: 内存占用
    test_memory_usage()

    print("\n" + "=" * 80)
    print("✅ 所有测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

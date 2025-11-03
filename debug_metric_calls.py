"""
超详细调试：逐步对比原始 vs 优化版本
"""

import torch
import numpy as np
import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "temp_aepsych"))

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


def test_metric_calls():
    """测试 _metric 调用的一致性"""
    print("=" * 80)
    print("测试：_metric 调用一致性")
    print("=" * 80)

    model = create_test_model()

    from extensions.dynamic_eur_acquisition.eur_anova_pair_acquisition import (
        EURAnovaPairAcqf,
    )

    config = {
        "gamma": 0.25,
        "main_weight": 1.0,
        "interaction_pairs": "0,1;2,3;4,5",
        "local_num": 4,
        "variable_types_list": "categorical, integer, integer, continuous, categorical, integer",
    }

    acqf = EURAnovaPairAcqf(model, **config)

    # 测试点
    torch.manual_seed(123)
    X_test = torch.rand(1, 6)

    # 获取canonicalized形式
    X_flat = X_test.reshape(-1, X_test.shape[-1])
    X_can_t = acqf._canonicalize_torch(X_flat)

    print(f"\n测试候选点: {X_can_t.numpy()}")
    print(f"B={X_can_t.shape[0]}, d={X_can_t.shape[1]}")

    # 基线信息
    I0 = acqf._metric(X_can_t)
    print(f"\n基线 I0: {I0.item():.8f}")

    # 主效应
    print(f"\n主效应（逐个维度）:")
    for i in range(6):
        X_i = acqf._make_local_hybrid(X_can_t, dims=[i])
        print(f"  X_{i} 形状: {X_i.shape}")
        Ii_raw = acqf._metric(X_i)
        print(f"  Ii_raw 形状: {Ii_raw.shape}, 值: {Ii_raw[:4].numpy()}")
        Ii = Ii_raw.view(1, 4).mean(dim=1)
        print(f"  Ii (平均后): {Ii.item():.8f}")
        Di = torch.clamp(Ii - I0, min=0.0)
        print(f"  Di (clamp后): {Di.item():.8f}")
        print()

    # 交互效应
    print(f"交互效应（对: {acqf._pairs}):")
    for i, j in acqf._pairs:
        X_ij = acqf._make_local_hybrid(X_can_t, dims=[i, j])
        print(f"  X_{i},{j} 形状: {X_ij.shape}")
        Iij_raw = acqf._metric(X_ij)
        print(f"  Iij_raw 形状: {Iij_raw.shape}, 值: {Iij_raw[:4].numpy()}")
        Iij = Iij_raw.view(1, 4).mean(dim=1)
        print(f"  Iij (平均后): {Iij.item():.8f}")
        print()


def test_batch_metric():
    """测试批量 _metric 调用"""
    print("\n" + "=" * 80)
    print("测试：批量 _metric 调用")
    print("=" * 80)

    model = create_test_model()

    from extensions.dynamic_eur_acquisition.eur_anova_pair_acquisition import (
        EURAnovaPairAcqf,
    )

    config = {
        "gamma": 0.25,
        "main_weight": 1.0,
        "interaction_pairs": "0,1;2,3;4,5",
        "local_num": 4,
        "variable_types_list": "categorical, integer, integer, continuous, categorical, integer",
    }

    acqf = EURAnovaPairAcqf(model, **config)

    # 测试点
    torch.manual_seed(123)
    X_test = torch.rand(1, 6)
    X_flat = X_test.reshape(-1, X_test.shape[-1])
    X_can_t = acqf._canonicalize_torch(X_flat)

    # 收集所有局部点
    X_all_local = []
    for i in range(6):
        X_i = acqf._make_local_hybrid(X_can_t, dims=[i])
        X_all_local.append(X_i)

    for i, j in acqf._pairs:
        X_ij = acqf._make_local_hybrid(X_can_t, dims=[i, j])
        X_all_local.append(X_ij)

    # 批量调用
    X_batch = torch.cat(X_all_local, dim=0)
    print(f"\n批量输入形状: {X_batch.shape}")
    print(
        f"  预期：{len(X_all_local)} 个段，每段 {1 * 4} 个点 = {len(X_all_local) * 4} 个点"
    )

    I_batch = acqf._metric(X_batch)
    print(f"\n批量输出形状: {I_batch.shape}")
    print(f"  前20个值: {I_batch[:20].numpy()}")

    # 逐段解包
    print(f"\n逐段解包:")
    current_row = 0
    for seg_idx in range(len(X_all_local)):
        seg_size = 1 * 4  # B * local_num
        start_row = current_row
        end_row = current_row + seg_size
        I_seg_raw = I_batch[start_row:end_row]
        I_seg = I_seg_raw.view(1, 4).mean(dim=1)

        print(
            f"  段 {seg_idx}: 行 [{start_row}:{end_row}], raw={I_seg_raw.numpy()}, mean={I_seg.item():.8f}"
        )
        current_row = end_row


if __name__ == "__main__":
    test_metric_calls()
    test_batch_metric()

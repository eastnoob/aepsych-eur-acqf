"""
增强测试：验证边界情况处理（问题1和问题2的修复）

测试目标：
1. 预计算失败场景（索引越界、空值、异常）
2. 分类维降级策略（保持原值 vs 非法值）
3. 警告机制（首次警告、汇总报告）
4. 核心功能不受影响
"""

import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
import warnings

from eur_anova_pair_acquisition import EURAnovaPairAcqf


def test_boundary_case_1_index_out_of_range():
    """测试场景1：variable_types 索引越界"""
    print("=" * 80)
    print("测试1: 分类维索引越界处理")
    print("=" * 80)

    X_train = torch.randn(20, 3, dtype=torch.float64)  # 只有3维
    y_train = torch.randn(20, 1, dtype=torch.float64)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())

    # 配置错误：索引5超出范围
    variable_types = {0: "continuous", 1: "continuous", 5: "categorical"}  # ❌ 越界

    print("\n【场景】variable_types 包含越界索引 5（数据只有3维）")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        acqf = EURAnovaPairAcqf(
            model=model, interaction_pairs=[(0, 1)], variable_types=variable_types
        )

        # 需要触发 _ensure_fresh_data 来执行预计算
        X_test = torch.randn(2, 1, 3, dtype=torch.float64)
        _ = acqf(X_test)

        # 检查警告
        precompute_warnings = [
            warning for warning in w if "预计算分类值失败" in str(warning.message)
        ]

        if len(precompute_warnings) > 0:
            print(f"  ✅ 正确捕获越界索引并警告:")
            print(f"     {precompute_warnings[0].message}")
        else:
            print(f"  ❌ 未捕获越界索引")
            return False

    # 验证字典中不包含失败的维度
    if 5 not in acqf._unique_vals_dict:
        print(f"  ✅ 越界维度未添加到字典")
    else:
        print(f"  ❌ 越界维度被错误添加")
        return False

    # 验证核心功能仍然正常
    X_test = torch.randn(5, 1, 3, dtype=torch.float64)
    try:
        acq_values = acqf(X_test)
        print(f"  ✅ Forward pass 成功（未受影响）")
        print(f"     输出形状: {acq_values.shape}")
    except Exception as e:
        print(f"  ❌ Forward pass 失败: {e}")
        return False

    print("\n✅ 测试1通过\n")
    return True


def test_boundary_case_2_empty_unique_values():
    """测试场景2：空 unique 值"""
    print("=" * 80)
    print("测试2: 空 unique 值处理")
    print("=" * 80)

    # 创建特殊数据：第2维所有值相同（导致只有1个unique值，测试单值情况）
    X_train = torch.randn(20, 3, dtype=torch.float64)
    y_train = torch.randn(20, 1, dtype=torch.float64)

    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())

    # 手动创建一个 acqf 并清空字典（模拟预计算失败）
    variable_types = {0: "continuous", 1: "continuous", 2: "categorical"}

    print("\n【场景】手动清空 unique_vals_dict（模拟预计算失败）")

    acqf = EURAnovaPairAcqf(
        model=model, interaction_pairs=[(0, 1)], variable_types=variable_types
    )

    # 人为清空字典模拟失败
    acqf._unique_vals_dict = {}

    # 测试降级行为
    print("\n【验证降级策略】")
    X_test = torch.randn(3, 1, 3, dtype=torch.float64)
    X_test[:, :, 2] = 1.0  # 设置原始值

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # 触发局部扰动
        X_local = acqf._make_local_hybrid(X_test.squeeze(1), dims=[2])

        # 检查降级警告
        fallback_warnings = [
            warning for warning in w if "保持原值" in str(warning.message)
        ]

        if len(fallback_warnings) > 0:
            print(f"  ✅ 正确发出降级警告:")
            print(f"     {fallback_warnings[0].message}")
        else:
            print(f"  ⚠️  未发出降级警告")

    # 验证值保持不变（降级策略）
    original_val = X_test[0, 0, 2].item()
    perturbed_vals = X_local[:, 2].cpu().numpy()

    if np.allclose(perturbed_vals, original_val):
        print(f"  ✅ 降级策略正确：保持原值 {original_val}")
    else:
        print(f"  ❌ 降级策略错误：期望 {original_val}，实际 {perturbed_vals[:3]}")
        return False

    print("\n✅ 测试2通过\n")
    return True


def test_boundary_case_3_warning_deduplication():
    """测试场景3：警告去重（避免重复警告）"""
    print("=" * 80)
    print("测试3: 警告去重机制")
    print("=" * 80)

    X_train = torch.randn(10, 3, dtype=torch.float64)
    y_train = torch.randn(10, 1, dtype=torch.float64)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())

    # 配置：分类维存在但unique值为空（模拟失败）
    acqf = EURAnovaPairAcqf(
        model=model,
        interaction_pairs=[(0, 1)],
        variable_types={0: "continuous", 1: "continuous", 2: "categorical"},
    )

    # 人为清空字典（模拟预计算失败）
    acqf._unique_vals_dict = {}

    print("\n【场景】多次调用 _make_local_hybrid（模拟500次采集）")

    X_test = torch.randn(5, 3, dtype=torch.float64)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # 模拟多次调用
        for i in range(10):
            _ = acqf._make_local_hybrid(X_test, dims=[2])

        # 统计降级警告数量
        fallback_warnings = [
            warning for warning in w if "保持原值" in str(warning.message)
        ]

        warning_count = len(fallback_warnings)

        if warning_count == 1:
            print(f"  ✅ 警告去重成功：10次调用仅警告1次")
        else:
            print(f"  ❌ 警告去重失败：10次调用警告{warning_count}次")
            return False

    print("\n✅ 测试3通过\n")
    return True


def test_normal_operation_unchanged():
    """测试场景4：正常操作完全不受影响"""
    print("=" * 80)
    print("测试4: 正常操作不受影响")
    print("=" * 80)

    # 正常配置
    X_train = torch.randn(30, 4, dtype=torch.float64)
    X_train[:, 2] = torch.randint(0, 3, (30,), dtype=torch.float64)  # 分类维
    y_train = torch.randn(30, 1, dtype=torch.float64)

    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())

    variable_types = {
        0: "continuous",
        1: "continuous",
        2: "categorical",  # 正常分类维
        3: "integer",
    }

    print("\n【场景】正常配置（无错误）")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        acqf = EURAnovaPairAcqf(
            model=model,
            interaction_pairs=[(0, 1), (1, 2)],
            variable_types=variable_types,
            gamma=0.3,
            main_weight=1.0,
            lambda_min=0.1,
            lambda_max=1.0,
        )

        # 不应有任何警告
        if len(w) == 0:
            print(f"  ✅ 无警告（正常运行）")
        else:
            print(f"  ⚠️  意外警告:")
            for warning in w:
                print(f"     {warning.message}")

    # 需要触发 _ensure_fresh_data 来执行预计算
    X_test_trigger = torch.randn(2, 1, 4, dtype=torch.float64)
    _ = acqf(X_test_trigger)

    # 验证预计算成功
    if 2 in acqf._unique_vals_dict:
        unique_vals = acqf._unique_vals_dict[2]
        print(f"  ✅ 分类维预计算成功: {len(unique_vals)} 个唯一值")
    else:
        print(f"  ❌ 分类维预计算失败")
        return False

    # 验证 forward pass
    X_test = torch.randn(5, 1, 4, dtype=torch.float64)

    try:
        acq_values = acqf(X_test)
        print(f"  ✅ Forward pass 成功")
        print(f"     输出形状: {acq_values.shape}")
        print(f"     采集值范围: [{acq_values.min():.4f}, {acq_values.max():.4f}]")

        if not torch.isnan(acq_values).any() and not torch.isinf(acq_values).any():
            print(f"  ✅ 无NaN/Inf值")
        else:
            print(f"  ❌ 包含NaN/Inf值")
            return False

    except Exception as e:
        print(f"  ❌ Forward pass 失败: {e}")
        return False

    # 验证分类维扰动正确
    print("\n【验证分类维扰动】")
    X_can = torch.randn(3, 4, dtype=torch.float64)
    X_can[:, 2] = 1.0  # 设置原始分类值

    X_local = acqf._make_local_hybrid(X_can, dims=[2])
    perturbed_vals = X_local[:, 2].cpu().numpy()

    # 检查是否都是合法值
    valid_vals = set(unique_vals)
    perturbed_unique = set(perturbed_vals)

    if perturbed_unique.issubset(valid_vals):
        print(f"  ✅ 扰动值都是合法分类值: {perturbed_unique}")
    else:
        print(f"  ❌ 扰动值包含非法值:")
        print(f"     合法: {valid_vals}")
        print(f"     实际: {perturbed_unique}")
        return False

    print("\n✅ 测试4通过\n")
    return True


def main():
    """运行所有边界测试"""
    print("\n" + "=" * 80)
    print("边界情况增强测试套件（问题1+问题2修复验证）")
    print("=" * 80 + "\n")

    results = []

    # 测试1: 索引越界
    results.append(("索引越界处理", test_boundary_case_1_index_out_of_range()))

    # 测试2: 空unique值
    results.append(("空unique值处理", test_boundary_case_2_empty_unique_values()))

    # 测试3: 警告去重
    results.append(("警告去重机制", test_boundary_case_3_warning_deduplication()))

    # 测试4: 正常操作
    results.append(("正常操作不受影响", test_normal_operation_unchanged()))

    # 汇总结果
    print("=" * 80)
    print("边界测试结果汇总")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12} {name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 所有边界测试通过！修复完美且安全！")
    else:
        print("❌ 部分测试失败，请检查修复代码")
    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

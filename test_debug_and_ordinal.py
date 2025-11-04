"""
测试调试工具和 ordinal 支持

测试内容：
1. 测试 get_diagnostics() 和 print_diagnostics() 功能
2. 检测 gower_distance 是否支持 ordinal 类型
"""

import torch
import numpy as np
import warnings

print("=" * 70)
print("测试1：调试工具功能")
print("=" * 70)


# 创建 mock 模型
class MockModel:
    def __init__(self):
        self.train_inputs = (torch.randn(15, 4),)  # 15个样本，4维
        self.train_targets = torch.randn(15)

    def posterior(self, X):
        class MockPosterior:
            def __init__(self, X):
                self.mean = torch.randn(X.shape[0], 1)
                self.variance = torch.ones(X.shape[0], 1) * 0.5

        return MockPosterior(X)


try:
    from eur_anova_pair import EURAnovaPairAcqf

    model = MockModel()

    # 测试1.1：不启用 debug_components
    print("\n【测试1.1】不启用 debug_components（默认静默）")
    print("-" * 70)
    acqf_no_debug = EURAnovaPairAcqf(
        model,
        total_budget=25,
        interaction_pairs=[(0, 1), (1, 2), (2, 3)],
    )

    X_test = torch.randn(10, 1, 4)
    scores = acqf_no_debug(X_test)

    print("✅ forward() 执行成功（无调试输出）")

    # 获取基本诊断信息（不需要 debug_components）
    diag = acqf_no_debug.get_diagnostics()
    print(f"✅ get_diagnostics() 成功")
    print(f"   - λ_t = {diag['lambda_t']:.3f}")
    print(f"   - γ_t = {diag['gamma_t']:.3f}")
    print(f"   - 训练样本数 = {diag['n_train']}")
    print(f"   - 交互对数量 = {diag['n_pairs']}")

    # 检查是否有效应贡献数据
    if "main_effects_sum" in diag:
        print("   ❌ 不应该有效应贡献数据（debug_components=False）")
    else:
        print("   ✅ 正确：无效应贡献数据（需要 debug_components=True）")

    # 测试1.2：启用 debug_components
    print("\n【测试1.2】启用 debug_components（按需查看详细信息）")
    print("-" * 70)
    acqf_with_debug = EURAnovaPairAcqf(
        model,
        total_budget=25,
        interaction_pairs=[(0, 1), (1, 2), (2, 3)],
        debug_components=True,  # 启用详细调试
    )

    scores = acqf_with_debug(X_test)
    print("✅ forward() 执行成功")

    # 获取详细诊断信息
    diag = acqf_with_debug.get_diagnostics()
    print(f"✅ get_diagnostics() 成功")

    if "main_effects_sum" in diag:
        print(f"   ✅ 包含效应贡献数据")
        print(f"      - 主效应形状: {diag['main_effects_sum'].shape}")
        print(f"      - 交互效应形状: {diag['pair_effects_sum'].shape}")
    else:
        print("   ❌ 应该有效应贡献数据")

    # 测试1.3：打印诊断信息（非 verbose）
    print("\n【测试1.3】打印诊断信息（简洁模式）")
    print("-" * 70)
    acqf_with_debug.print_diagnostics(verbose=False)

    # 测试1.4：打印诊断信息（verbose）
    print("\n【测试1.4】打印诊断信息（详细模式）")
    print("-" * 70)
    acqf_with_debug.print_diagnostics(verbose=True)

    print("\n✅ 调试工具测试全部通过！")

except Exception as e:
    print(f"❌ 调试工具测试失败: {e}")
    import traceback

    traceback.print_exc()


# ========================================================================
# 测试2：检测 gower_distance 是否支持 ordinal
# ========================================================================

print("\n\n" + "=" * 70)
print("测试2：检测 ordinal 支持情况")
print("=" * 70)

try:
    from gower_distance import compute_coverage_batch

    # 创建测试数据
    X_hist = np.array(
        [
            [1, 0, 5.0],
            [2, 1, 3.5],
            [3, 0, 7.2],
            [1, 1, 2.1],
        ]
    )

    X_cand = np.array(
        [
            [2, 0, 4.0],
            [1, 1, 6.5],
        ]
    )

    print("\n【测试2.1】支持的变量类型检测")
    print("-" * 70)

    # 测试基本类型（应该支持）
    test_cases = [
        ("categorical", {"categorical"}),
        ("continuous", {"continuous"}),
        ("mixed", {"categorical", "continuous"}),
    ]

    supported_types = set()

    for test_name, types_to_test in test_cases:
        try:
            if "categorical" in types_to_test and "continuous" in types_to_test:
                vt = {0: "categorical", 1: "categorical", 2: "continuous"}
            elif "categorical" in types_to_test:
                vt = {0: "categorical", 1: "categorical", 2: "categorical"}
            else:
                vt = {0: "continuous", 1: "continuous", 2: "continuous"}

            cov = compute_coverage_batch(X_cand, X_hist, variable_types=vt)

            print(f"✅ {test_name:15s}: 支持")
            supported_types.update(types_to_test)

        except Exception as e:
            print(f"❌ {test_name:15s}: 不支持 ({str(e)[:50]})")

    # 测试 ordinal 类型
    print("\n【测试2.2】ordinal 类型支持检测")
    print("-" * 70)

    ordinal_test_cases = [
        ("ordinal_only", {0: "ordinal", 1: "ordinal", 2: "ordinal"}),
        ("ordinal_mixed", {0: "ordinal", 1: "categorical", 2: "continuous"}),
        ("integer_as_ordinal", {0: "integer", 1: "integer", 2: "continuous"}),
    ]

    ordinal_supported = False

    for test_name, vt in ordinal_test_cases:
        try:
            cov = compute_coverage_batch(X_cand, X_hist, variable_types=vt)
            print(f"✅ {test_name:20s}: 支持！")
            ordinal_supported = True

        except (ValueError, KeyError, TypeError) as e:
            error_msg = str(e)
            if "ordinal" in error_msg.lower() or "integer" in error_msg.lower():
                print(f"❌ {test_name:20s}: 不支持（明确拒绝）")
            else:
                print(f"❌ {test_name:20s}: 不支持（{error_msg[:40]}...）")

        except Exception as e:
            print(f"⚠️  {test_name:20s}: 未知错误 ({str(e)[:40]}...)")

    # 测试降级策略
    print("\n【测试2.3】降级策略验证")
    print("-" * 70)

    try:
        # 模拟当前代码的降级逻辑
        user_types = {0: "ordinal", 1: "integer", 2: "continuous"}
        downgraded_types = {
            k: ("categorical" if v == "categorical" else "continuous")
            for k, v in user_types.items()
        }

        print(f"用户定义类型: {user_types}")
        print(f"降级后类型:   {downgraded_types}")

        cov = compute_coverage_batch(X_cand, X_hist, variable_types=downgraded_types)

        print(f"✅ 降级策略有效（ordinal/integer → continuous）")
        print(f"   覆盖度结果: {cov}")

    except Exception as e:
        print(f"❌ 降级策略失败: {e}")

    # 总结
    print("\n【测试结果总结】")
    print("-" * 70)
    print(f"支持的基本类型: {sorted(supported_types)}")

    if ordinal_supported:
        print("✅ ordinal 类型: 原生支持")
        print("   建议: 可以在代码中直接使用 'ordinal'")
    else:
        print("❌ ordinal 类型: 不支持")
        print("   建议: 保持当前降级策略（ordinal/integer → continuous）")
        print("   影响: 对于等距整数变量（如 Likert 量表），影响很小")

except ImportError as e:
    print(f"❌ 无法导入 gower_distance: {e}")
    print("   请确保文件路径正确")

except Exception as e:
    print(f"❌ ordinal 支持检测失败: {e}")
    import traceback

    traceback.print_exc()


print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)

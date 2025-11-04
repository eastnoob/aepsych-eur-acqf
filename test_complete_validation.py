"""
完整功能验证测试

测试以下方面：
1. ✅ total_budget 自适应配置优先级（6个场景）
2. ✅ 参数验证逻辑（防止错误配置）
3. ✅ 边界值处理
4. ✅ 与现有功能的兼容性
"""

import warnings
from unittest.mock import Mock
from eur_anova_pair import EURAnovaPairAcqf


def create_mock_model():
    """创建简单的mock模型"""
    model = Mock()
    model.train_inputs = (Mock(),)
    model.train_inputs[0].device = Mock()
    model.train_inputs[0].dtype = Mock()
    return model


def test_parameter_validation():
    """测试参数验证逻辑"""
    print("\n" + "=" * 70)
    print("测试组1: 参数验证逻辑")
    print("=" * 70)

    model = create_mock_model()

    # 测试1: tau_n_max <= tau_n_min 应该报错
    print("\n  测试1.1: tau_n_max <= tau_n_min 应该抛出 ValueError")
    try:
        acqf = EURAnovaPairAcqf(
            model=model,
            tau_n_min=10,
            tau_n_max=5,  # 错误：小于 tau_n_min
            variable_types={0: "continuous"},
        )
        print("  ❌ 应该抛出 ValueError，但没有")
        assert False
    except ValueError as e:
        assert "tau_n_max must be > tau_n_min" in str(e)
        print(f"  ✅ 正确抛出 ValueError: {e}")

    # 测试2: gamma_max < gamma_min 应该报错
    print("\n  测试1.2: gamma_max < gamma_min 应该抛出 ValueError")
    try:
        acqf = EURAnovaPairAcqf(
            model=model,
            gamma_min=0.3,
            gamma_max=0.1,  # 错误：小于 gamma_min
            variable_types={0: "continuous"},
        )
        print("  ❌ 应该抛出 ValueError，但没有")
        assert False
    except ValueError as e:
        assert "gamma_max must be >= gamma_min" in str(e)
        print(f"  ✅ 正确抛出 ValueError: {e}")

    # 测试3: lambda_max < lambda_min 应该报错
    print("\n  测试1.3: lambda_max < lambda_min 应该抛出 ValueError")
    try:
        acqf = EURAnovaPairAcqf(
            model=model,
            lambda_min=0.8,
            lambda_max=0.3,  # 错误：小于 lambda_min
            variable_types={0: "continuous"},
        )
        print("  ❌ 应该抛出 ValueError，但没有")
        assert False
    except ValueError as e:
        assert "lambda_max must be >= lambda_min" in str(e)
        print(f"  ✅ 正确抛出 ValueError: {e}")

    # 测试4: tau1 <= tau2 应该报错
    print("\n  测试1.4: tau1 <= tau2 应该抛出 ValueError")
    try:
        acqf = EURAnovaPairAcqf(
            model=model,
            tau1=5,  # 错误：小于等于 tau2
            tau2=10,
            variable_types={0: "continuous"},
        )
        print("  ❌ 应该抛出 ValueError，但没有")
        assert False
    except ValueError as e:
        assert "tau1 must be > tau2" in str(e)
        print(f"  ✅ 正确抛出 ValueError: {e}")

    # 测试5: main_weight <= 0 应该报错
    print("\n  测试1.5: main_weight <= 0 应该抛出 ValueError")
    try:
        acqf = EURAnovaPairAcqf(
            model=model, main_weight=0, variable_types={0: "continuous"}
        )
        print("  ❌ 应该抛出 ValueError，但没有")
        assert False
    except ValueError as e:
        assert "main_weight must be positive" in str(e)
        print(f"  ✅ 正确抛出 ValueError: {e}")

    print("\n✅ 参数验证测试全部通过")


def test_total_budget_priority():
    """测试 total_budget 优先级逻辑（核心功能）"""
    print("\n" + "=" * 70)
    print("测试组2: total_budget 自适应配置优先级")
    print("=" * 70)

    model = create_mock_model()

    # 场景1: 只提供 total_budget
    print("\n  场景2.1: 只提供 total_budget=50")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(
            model=model, total_budget=50, variable_types={0: "continuous"}
        )
        assert acqf.tau_n_max == 35, f"期望35，实际{acqf.tau_n_max}"
        assert acqf.gamma_min == 0.1, f"期望0.1，实际{acqf.gamma_min}"
        assert len(w) >= 2, "应该有自适应警告"
        print(f"  ✅ tau_n_max={acqf.tau_n_max}, gamma_min={acqf.gamma_min}")

    # 场景2: 手动配置 + total_budget（手动优先）
    print("\n  场景2.2: 手动配置 + total_budget（手动应该优先）")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(
            model=model,
            tau_n_max=40,
            gamma_min=0.08,
            total_budget=50,
            variable_types={0: "continuous"},
        )
        assert acqf.tau_n_max == 40, f"期望40，实际{acqf.tau_n_max}"
        assert acqf.gamma_min == 0.08, f"期望0.08，实际{acqf.gamma_min}"
        adapt_warnings = [x for x in w if "实验预算自适应" in str(x.message)]
        assert len(adapt_warnings) == 0, "不应有自适应警告"
        print(f"  ✅ tau_n_max={acqf.tau_n_max}, gamma_min={acqf.gamma_min}")

    # 场景3: 手动配置=默认值 + total_budget（关键测试）
    print("\n  场景2.3: 手动配置=默认值 + total_budget（关键：手动仍应优先）")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(
            model=model,
            tau_n_max=25,  # 恰好等于默认值
            total_budget=50,  # 会想设为35
            variable_types={0: "continuous"},
        )
        assert acqf.tau_n_max == 25, f"期望25（手动），实际{acqf.tau_n_max}"
        assert acqf.gamma_min == 0.1, f"期望0.1（自适应），实际{acqf.gamma_min}"
        adapt_warnings = [x for x in w if "实验预算自适应" in str(x.message)]
        assert len(adapt_warnings) == 1, "应该只有gamma_min的警告"
        print(f"  ✅ tau_n_max={acqf.tau_n_max}（保持手动值，未被覆盖）")
        print(f"  ✅ gamma_min={acqf.gamma_min}（自适应生效）")

    # 场景4: 什么都不配置（默认值）
    print("\n  场景2.4: 什么都不配置（应使用默认值）")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(model=model, variable_types={0: "continuous"})
        assert acqf.tau_n_max == 25, f"期望25（默认），实际{acqf.tau_n_max}"
        assert acqf.gamma_min == 0.05, f"期望0.05（默认），实际{acqf.gamma_min}"
        adapt_warnings = [x for x in w if "实验预算自适应" in str(x.message)]
        assert len(adapt_warnings) == 0, "不应有自适应警告"
        print(f"  ✅ tau_n_max={acqf.tau_n_max}, gamma_min={acqf.gamma_min}")

    print("\n✅ total_budget 优先级测试全部通过")


def test_boundary_conditions():
    """测试边界条件"""
    print("\n" + "=" * 70)
    print("测试组3: 边界条件")
    print("=" * 70)

    model = create_mock_model()

    # 测试1: total_budget 边界（30）
    print("\n  测试3.1: total_budget=30 (gamma_min边界)")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(
            model=model, total_budget=30, variable_types={0: "continuous"}
        )
        assert acqf.gamma_min == 0.1, f"期望0.1 (>=30)，实际{acqf.gamma_min}"
        print(f"  ✅ gamma_min={acqf.gamma_min} (正确应用边界规则)")

    # 测试2: total_budget < 30
    print("\n  测试3.2: total_budget=20 (<30)")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(
            model=model, total_budget=20, variable_types={0: "continuous"}
        )
        assert acqf.gamma_min == 0.05, f"期望0.05 (<30)，实际{acqf.gamma_min}"
        assert acqf.tau_n_max == 14, f"期望14 (20*0.7)，实际{acqf.tau_n_max}"
        print(f"  ✅ gamma_min={acqf.gamma_min}, tau_n_max={acqf.tau_n_max} (正确计算)")

    # 测试3: 极小值
    print("\n  测试3.3: tau_n_min=1, tau_n_max=2 (最小间隔)")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(
            model=model, tau_n_min=1, tau_n_max=2, variable_types={0: "continuous"}
        )
        assert acqf.tau_n_min == 1
        assert acqf.tau_n_max == 2
        print(f"  ✅ 接受最小间隔配置")

    # 测试4: gamma边界（相等）
    print("\n  测试3.4: gamma_min=gamma_max (边界相等)")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(
            model=model,
            gamma_min=0.2,
            gamma_max=0.2,
            variable_types={0: "continuous"},
        )
        assert acqf.gamma_min == 0.2
        assert acqf.gamma_max == 0.2
        print(f"  ✅ 接受相等边界")

    print("\n✅ 边界条件测试全部通过")


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n" + "=" * 70)
    print("测试组4: 向后兼容性")
    print("=" * 70)

    model = create_mock_model()

    # 测试1: 旧版本配置方式（不使用total_budget）
    print("\n  测试4.1: 旧版本配置方式（完全手动）")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(
            model=model,
            tau_n_min=5,
            tau_n_max=30,
            gamma_min=0.06,
            gamma_max=0.4,
            lambda_min=0.1,
            lambda_max=0.9,
            tau1=15,
            tau2=8,
            variable_types={0: "continuous", 1: "continuous"},
        )
        # 验证所有参数正确设置
        assert acqf.tau_n_min == 5
        assert acqf.tau_n_max == 30
        assert acqf.gamma_min == 0.06
        assert acqf.gamma_max == 0.4
        assert acqf.lambda_min == 0.1
        assert acqf.lambda_max == 0.9
        assert acqf.tau1 == 15
        assert acqf.tau2 == 8
        print("  ✅ 旧版本配置完全兼容")

    # 测试2: 混合使用（部分新功能）
    print("\n  测试4.2: 混合使用新旧功能")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(
            model=model,
            tau_n_max=35,  # 手动配置
            total_budget=50,  # 新功能（但不影响tau_n_max）
            variable_types={0: "continuous"},
        )
        assert acqf.tau_n_max == 35  # 手动配置保持
        assert acqf.gamma_min == 0.1  # 自适应生效
        print("  ✅ 新旧功能混合使用正常")

    print("\n✅ 向后兼容性测试全部通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🔬 开始完整功能验证".center(70, "="))

    try:
        test_parameter_validation()  # 测试组1
        test_total_budget_priority()  # 测试组2（核心）
        test_boundary_conditions()  # 测试组3
        test_backward_compatibility()  # 测试组4

        print("\n" + "=" * 70)
        print("🎉 所有测试通过！功能完美符合预期")
        print("=" * 70)
        print("\n验证结果总结：")
        print("  ✅ 测试组1: 参数验证逻辑（5个测试）")
        print("  ✅ 测试组2: total_budget 优先级（4个场景）")
        print("  ✅ 测试组3: 边界条件（4个测试）")
        print("  ✅ 测试组4: 向后兼容性（2个测试）")
        print("\n核心改进验证：")
        print("  🎯 使用 None 哨兵值正确区分手动/自动配置")
        print("  🎯 手动配置=默认值时，不会被自适应错误覆盖")
        print("  🎯 完全向后兼容，旧代码无需修改")
        print("  🎯 所有参数验证逻辑正确工作")

        return True

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

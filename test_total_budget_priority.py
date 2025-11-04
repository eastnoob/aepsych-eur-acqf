"""
测试 total_budget 自适应配置的优先级逻辑

验证以下场景：
1. 只提供 total_budget（自适应生效）
2. 手动配置 + total_budget（手动优先）
3. 手动配置恰好等于默认值 + total_budget（手动仍然优先）
4. 只手动配置（使用手动值）
5. 什么都不配置（使用默认值）
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


def test_scenario_1_only_total_budget():
    """场景1: 只提供 total_budget，应该自适应配置"""
    print("\n" + "=" * 70)
    print("场景1: 只提供 total_budget=50")
    print("=" * 70)

    model = create_mock_model()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        acqf = EURAnovaPairAcqf(
            model=model,
            total_budget=50,
            variable_types={0: "continuous", 1: "continuous"},
        )

        # 验证自适应生效
        assert (
            acqf.tau_n_max == 35
        ), f"期望 tau_n_max=35 (50*0.7)，实际={acqf.tau_n_max}"
        assert (
            acqf.gamma_min == 0.1
        ), f"期望 gamma_min=0.1 (budget>=30)，实际={acqf.gamma_min}"

        # 验证有警告
        assert len(w) == 2, f"应该有2个警告，实际有{len(w)}个"

        print(f"✅ tau_n_max = {acqf.tau_n_max} (自适应: 50 * 0.7 = 35)")
        print(f"✅ gamma_min = {acqf.gamma_min} (自适应: budget>=30 → 0.1)")
        print(f"✅ 发出了 {len(w)} 个自适应警告")


def test_scenario_2_manual_override():
    """场景2: 手动配置 + total_budget，手动配置应该优先"""
    print("\n" + "=" * 70)
    print("场景2: 手动配置 tau_n_max=40, gamma_min=0.08 + total_budget=50")
    print("=" * 70)

    model = create_mock_model()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        acqf = EURAnovaPairAcqf(
            model=model,
            tau_n_max=40,  # 手动配置
            gamma_min=0.08,  # 手动配置
            total_budget=50,
            variable_types={0: "continuous", 1: "continuous"},
        )

        # 验证手动配置保持不变
        assert acqf.tau_n_max == 40, f"期望 tau_n_max=40 (手动)，实际={acqf.tau_n_max}"
        assert (
            acqf.gamma_min == 0.08
        ), f"期望 gamma_min=0.08 (手动)，实际={acqf.gamma_min}"

        # 验证没有自适应警告
        adapt_warnings = [x for x in w if "实验预算自适应" in str(x.message)]
        assert (
            len(adapt_warnings) == 0
        ), f"不应有自适应警告，但有{len(adapt_warnings)}个"

        print(f"✅ tau_n_max = {acqf.tau_n_max} (保持手动配置，未被 total_budget 覆盖)")
        print(f"✅ gamma_min = {acqf.gamma_min} (保持手动配置，未被 total_budget 覆盖)")
        print(f"✅ 没有发出自适应警告（手动配置优先）")


def test_scenario_3_manual_equals_default():
    """场景3: 手动配置恰好等于默认值 + total_budget，手动配置仍然优先"""
    print("\n" + "=" * 70)
    print("场景3: 手动配置 tau_n_max=25 (恰好是默认值) + total_budget=50")
    print("=" * 70)

    model = create_mock_model()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        acqf = EURAnovaPairAcqf(
            model=model,
            tau_n_max=25,  # 手动配置，恰好等于默认值
            total_budget=50,  # 自适应会想设为35
            variable_types={0: "continuous", 1: "continuous"},
        )

        # ✅ 关键测试：即使手动值=默认值，也应该保持25，不被自适应覆盖
        assert acqf.tau_n_max == 25, f"期望 tau_n_max=25 (手动)，实际={acqf.tau_n_max}"

        # gamma_min 未手动配置，应该被自适应
        assert (
            acqf.gamma_min == 0.1
        ), f"期望 gamma_min=0.1 (自适应)，实际={acqf.gamma_min}"

        # 只应该有1个警告（gamma_min的自适应）
        adapt_warnings = [x for x in w if "实验预算自适应" in str(x.message)]
        assert (
            len(adapt_warnings) == 1
        ), f"应该有1个自适应警告（gamma_min），实际有{len(adapt_warnings)}个"

        print(f"✅ tau_n_max = {acqf.tau_n_max} (保持手动值25，未被自适应改为35)")
        print(f"✅ gamma_min = {acqf.gamma_min} (自适应生效)")
        print(f"✅ 只有1个自适应警告（仅 gamma_min）")
        print("   🎯 关键验证：手动配置即使等于默认值，也不会被自适应覆盖")


def test_scenario_4_only_manual():
    """场景4: 只手动配置，没有 total_budget"""
    print("\n" + "=" * 70)
    print("场景4: 只手动配置 tau_n_max=30, gamma_min=0.06")
    print("=" * 70)

    model = create_mock_model()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        acqf = EURAnovaPairAcqf(
            model=model,
            tau_n_max=30,
            gamma_min=0.06,
            variable_types={0: "continuous", 1: "continuous"},
        )

        # 验证使用手动值
        assert acqf.tau_n_max == 30, f"期望 tau_n_max=30，实际={acqf.tau_n_max}"
        assert acqf.gamma_min == 0.06, f"期望 gamma_min=0.06，实际={acqf.gamma_min}"

        # 验证没有自适应警告
        adapt_warnings = [x for x in w if "实验预算自适应" in str(x.message)]
        assert len(adapt_warnings) == 0, f"不应有自适应警告"

        print(f"✅ tau_n_max = {acqf.tau_n_max} (使用手动配置)")
        print(f"✅ gamma_min = {acqf.gamma_min} (使用手动配置)")
        print(f"✅ 没有自适应警告")


def test_scenario_5_all_default():
    """场景5: 什么都不配置，使用默认值"""
    print("\n" + "=" * 70)
    print("场景5: 不提供任何配置")
    print("=" * 70)

    model = create_mock_model()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        acqf = EURAnovaPairAcqf(
            model=model, variable_types={0: "continuous", 1: "continuous"}
        )

        # 验证使用默认值
        assert acqf.tau_n_max == 25, f"期望 tau_n_max=25 (默认)，实际={acqf.tau_n_max}"
        assert (
            acqf.gamma_min == 0.05
        ), f"期望 gamma_min=0.05 (默认)，实际={acqf.gamma_min}"

        # 验证没有自适应警告
        adapt_warnings = [x for x in w if "实验预算自适应" in str(x.message)]
        assert len(adapt_warnings) == 0, f"不应有自适应警告"

        print(f"✅ tau_n_max = {acqf.tau_n_max} (默认值)")
        print(f"✅ gamma_min = {acqf.gamma_min} (默认值)")
        print(f"✅ 没有自适应警告")


def test_scenario_6_edge_cases():
    """场景6: 边界情况测试"""
    print("\n" + "=" * 70)
    print("场景6: 边界情况测试")
    print("=" * 70)

    model = create_mock_model()

    # 测试 total_budget < 30
    print("\n  子场景6.1: total_budget=20 (< 30)")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(
            model=model, total_budget=20, variable_types={0: "continuous"}
        )
        assert (
            acqf.tau_n_max == 14
        ), f"期望 tau_n_max=14 (20*0.7)，实际={acqf.tau_n_max}"
        assert (
            acqf.gamma_min == 0.05
        ), f"期望 gamma_min=0.05 (budget<30)，实际={acqf.gamma_min}"
        print(f"  ✅ tau_n_max = {acqf.tau_n_max}, gamma_min = {acqf.gamma_min}")

    # 测试 total_budget = 30 (边界)
    print("\n  子场景6.2: total_budget=30 (边界)")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(
            model=model, total_budget=30, variable_types={0: "continuous"}
        )
        assert (
            acqf.tau_n_max == 21
        ), f"期望 tau_n_max=21 (30*0.7)，实际={acqf.tau_n_max}"
        assert (
            acqf.gamma_min == 0.1
        ), f"期望 gamma_min=0.1 (budget>=30)，实际={acqf.gamma_min}"
        print(f"  ✅ tau_n_max = {acqf.tau_n_max}, gamma_min = {acqf.gamma_min}")

    # 测试部分手动配置
    print("\n  子场景6.3: 只手动配置 tau_n_max + total_budget")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        acqf = EURAnovaPairAcqf(
            model=model,
            tau_n_max=50,  # 手动
            total_budget=100,  # gamma_min 会被自适应
            variable_types={0: "continuous"},
        )
        assert acqf.tau_n_max == 50, f"期望 tau_n_max=50 (手动)，实际={acqf.tau_n_max}"
        assert (
            acqf.gamma_min == 0.1
        ), f"期望 gamma_min=0.1 (自适应)，实际={acqf.gamma_min}"
        adapt_warnings = [x for x in w if "实验预算自适应" in str(x.message)]
        assert len(adapt_warnings) == 1, "应该只有gamma_min的自适应警告"
        print(
            f"  ✅ tau_n_max = {acqf.tau_n_max} (手动), gamma_min = {acqf.gamma_min} (自适应)"
        )


if __name__ == "__main__":
    print("\n" + "🔍 开始测试 total_budget 自适应配置优先级".center(70, "="))

    try:
        test_scenario_1_only_total_budget()
        test_scenario_2_manual_override()
        test_scenario_3_manual_equals_default()  # 🎯 关键测试
        test_scenario_4_only_manual()
        test_scenario_5_all_default()
        test_scenario_6_edge_cases()

        print("\n" + "=" * 70)
        print("✅ 所有测试通过！total_budget 自适应配置逻辑正确")
        print("=" * 70)
        print("\n关键验证点：")
        print("  1. ✅ 只提供 total_budget 时，自适应生效")
        print("  2. ✅ 手动配置 + total_budget 时，手动配置优先")
        print("  3. ✅ 手动配置=默认值时，仍然保持手动值（不被自适应覆盖）")
        print("  4. ✅ 只手动配置时，使用手动值")
        print("  5. ✅ 都不配置时，使用默认值")
        print("  6. ✅ 边界情况和部分配置正确处理")
        print("\n🎯 核心改进：使用 None 哨兵值正确区分手动/自动配置")
        print("   而非通过 '值是否等于默认值' 判断（避免误判）")

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback

        traceback.print_exc()
        raise

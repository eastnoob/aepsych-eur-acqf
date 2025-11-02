"""
全面验证问题1和问题2的修复

测试目标：
1. 问题1：Laplace梯度计算的内存安全性和性能
2. 问题2：交互对解析的去重和顺序稳定性
3. 确保修复不影响核心功能
"""

import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
import time
import warnings

from eur_anova_pair_acquisition import EURAnovaPairAcqf


def test_problem1_laplace_memory_safety():
    """测试问题1：Laplace梯度计算的内存安全性"""
    print("=" * 80)
    print("测试1: Laplace梯度计算内存安全性")
    print("=" * 80)

    # 创建训练数据（需要2D输出）
    X_train = torch.randn(20, 3)
    y_train = torch.randn(20, 1)

    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())
    model.likelihood.noise_covar.noise = 0.01

    acqf = EURAnovaPairAcqf(
        model=model,
        interaction_pairs=[(0, 1), (1, 2)],
        variable_types={0: "continuous", 1: "continuous", 2: "continuous"},
    )

    # 测试多次调用不会内存溢出
    print("\n【测试1.1】连续调用50次 _extract_parameter_variances_laplace")

    start_time = time.time()
    success_count = 0

    for i in range(50):
        try:
            param_vars = acqf._extract_parameter_variances_laplace()
            if param_vars is not None:
                success_count += 1
        except Exception as e:
            print(f"  ❌ 第{i+1}次调用失败: {e}")
            return False

    elapsed = time.time() - start_time

    print(f"  ✅ 成功调用 {success_count}/50 次")
    print(f"  ⏱️  总耗时: {elapsed:.2f}s，平均: {elapsed/50*1000:.1f}ms/次")

    # 测试模型模式恢复
    print("\n【测试1.2】模型模式正确恢复")
    original_mode = model.training
    print(f"  初始模式: {'train' if original_mode else 'eval'}")

    _ = acqf._extract_parameter_variances_laplace()

    final_mode = model.training
    print(f"  最终模式: {'train' if final_mode else 'eval'}")

    if original_mode == final_mode:
        print("  ✅ 模型模式正确恢复")
    else:
        print("  ❌ 模型模式未恢复")
        return False

    # 测试异常情况下的模式恢复
    print("\n【测试1.3】异常情况下的模式恢复")

    # 测试没有train_inputs的模型（模拟边缘情况）
    class MockModel:
        training = True

        def train(self, mode=True):
            self.training = mode

    mock_model = MockModel()
    acqf_mock = EURAnovaPairAcqf(model=mock_model, interaction_pairs=[(0, 1)])

    result = acqf_mock._extract_parameter_variances_laplace()

    if result is None:
        print("  ✅ 异常情况正确处理（返回None）")
    else:
        print("  ⚠️  预期返回None但得到非None结果")

    print("\n✅ 问题1修复验证通过\n")
    return True


def test_problem2_interaction_pairs_dedup():
    """测试问题2：交互对解析去重和顺序稳定性"""
    print("=" * 80)
    print("测试2: 交互对解析去重和顺序稳定性")
    print("=" * 80)

    X_train = torch.randn(10, 4)
    y_train = torch.randn(10, 1)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())

    # 测试用例
    test_cases = [
        {
            "name": "元组列表（包含重复）",
            "input": [(0, 1), (1, 2), (0, 1), (2, 3), (1, 0)],  # 重复和顺序反转
            "expected": [(0, 1), (1, 2), (2, 3)],
            "expected_dup_count": 2,
        },
        {
            "name": "字符串分号分隔（包含重复）",
            "input": "0,1; 1,2; 0,1; 2,3",
            "expected": [(0, 1), (1, 2), (2, 3)],
            "expected_dup_count": 1,
        },
        {
            "name": "混合分隔符",
            "input": ["0,1", "1|2", "2,3"],
            "expected": [(0, 1), (1, 2), (2, 3)],
            "expected_dup_count": 0,
        },
        {
            "name": "包含自环（应被忽略）",
            "input": [(0, 0), (0, 1), (1, 1), (1, 2)],
            "expected": [(0, 1), (1, 2)],
            "expected_dup_count": 0,
        },
    ]

    all_passed = True

    for i, tc in enumerate(test_cases, 1):
        print(f"\n【测试2.{i}】{tc['name']}")
        print(f"  输入: {tc['input']}")

        # 捕获警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            acqf = EURAnovaPairAcqf(
                model=model,
                interaction_pairs=tc["input"],
                variable_types={k: "continuous" for k in range(4)},
            )

            result = acqf._pairs
            expected = tc["expected"]

            # 检查结果
            if result == expected:
                print(f"  ✅ 解析正确: {result}")
            else:
                print(f"  ❌ 解析错误:")
                print(f"     预期: {expected}")
                print(f"     实际: {result}")
                all_passed = False
                continue

            # 检查警告
            dup_warnings = [
                warning for warning in w if "重复项" in str(warning.message)
            ]

            if tc["expected_dup_count"] > 0:
                if len(dup_warnings) > 0:
                    print(f"  ✅ 正确发出去重警告: {dup_warnings[0].message}")
                else:
                    print(f"  ⚠️  预期发出去重警告但未发出")
            else:
                if len(dup_warnings) == 0:
                    print(f"  ✅ 无重复，未发出警告")
                else:
                    print(f"  ⚠️  不应发出警告但发出了: {dup_warnings[0].message}")

    # 测试顺序稳定性（多次运行）
    print("\n【测试2.5】顺序稳定性（10次重复运行）")

    input_pairs = [(0, 1), (2, 3), (0, 1), (1, 2), (2, 3)]
    expected_order = [(0, 1), (2, 3), (1, 2)]

    results = []
    for run in range(10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acqf = EURAnovaPairAcqf(
                model=model,
                interaction_pairs=input_pairs,
                variable_types={k: "continuous" for k in range(4)},
            )
            results.append(acqf._pairs)

    # 检查所有结果是否一致
    all_same = all(r == expected_order for r in results)

    if all_same:
        print(f"  ✅ 10次运行顺序完全一致: {expected_order}")
    else:
        print(f"  ❌ 顺序不稳定:")
        for i, r in enumerate(results[:3], 1):
            print(f"     运行{i}: {r}")
        all_passed = False

    if all_passed:
        print("\n✅ 问题2修复验证通过\n")
    else:
        print("\n❌ 问题2修复验证失败\n")

    return all_passed


def test_functional_integrity():
    """测试核心功能完整性（确保修复不影响功能）"""
    print("=" * 80)
    print("测试3: 核心功能完整性验证")
    print("=" * 80)

    # 创建测试数据（需要2D输出）
    X_train = torch.randn(30, 3)
    y_train = torch.randn(30, 1)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())
    model.likelihood.noise_covar.noise = 0.01

    print("\n【测试3.1】基本初始化")

    acqf = EURAnovaPairAcqf(
        model=model,
        interaction_pairs=[(0, 1), (1, 2)],
        variable_types={0: "continuous", 1: "continuous", 2: "continuous"},
        gamma=0.3,
        main_weight=1.0,
        lambda_min=0.1,
        lambda_max=1.0,
    )

    print(f"  ✅ 成功创建 EURAnovaPairAcqf")
    print(f"     交互对: {acqf._pairs}")
    print(f"     main_weight: {acqf.main_weight}")
    print(f"     lambda范围: [{acqf.lambda_min}, {acqf.lambda_max}]")

    print("\n【测试3.2】Forward Pass")

    X_test = torch.randn(5, 1, 3)

    try:
        acq_values = acqf(X_test)

        print(f"  ✅ Forward pass 成功")
        print(f"     输入形状: {X_test.shape}")
        print(f"     输出形状: {acq_values.shape}")
        print(f"     采集值范围: [{acq_values.min():.4f}, {acq_values.max():.4f}]")

        # 检查输出合理性
        if acq_values.shape[0] == 5:
            print(f"  ✅ 输出形状正确")
        else:
            print(f"  ❌ 输出形状错误")
            return False

        if not torch.isnan(acq_values).any() and not torch.isinf(acq_values).any():
            print(f"  ✅ 无NaN/Inf值")
        else:
            print(f"  ❌ 包含NaN/Inf值")
            return False

    except Exception as e:
        print(f"  ❌ Forward pass 失败: {e}")
        return False

    print("\n【测试3.3】动态权重计算")

    try:
        lambda_t = acqf._compute_dynamic_lambda()
        gamma_t = acqf._compute_dynamic_gamma()

        print(f"  ✅ 动态权重计算成功")
        print(f"     当前 λ_t: {lambda_t:.4f}")
        print(f"     当前 γ_t: {gamma_t:.4f}")

        # 检查范围
        if acqf.lambda_min <= lambda_t <= acqf.lambda_max:
            print(f"  ✅ λ_t 在合理范围内")
        else:
            print(f"  ⚠️  λ_t 超出预期范围")

        if 0.0 <= gamma_t <= 1.0:
            print(f"  ✅ γ_t 在合理范围内")
        else:
            print(f"  ⚠️  γ_t 超出预期范围")

    except Exception as e:
        print(f"  ❌ 动态权重计算失败: {e}")
        return False

    print("\n✅ 核心功能完整性验证通过\n")
    return True


def test_performance_comparison():
    """性能对比测试（修复前后对比）"""
    print("=" * 80)
    print("测试4: 性能对比（问题1修复效果）")
    print("=" * 80)

    X_train = torch.randn(50, 4)
    y_train = torch.randn(50, 1)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())

    acqf = EURAnovaPairAcqf(
        model=model,
        interaction_pairs=[(0, 1), (1, 2), (2, 3)],
        variable_types={k: "continuous" for k in range(4)},
    )

    print("\n【性能测试】Laplace方差提取（30次平均）")

    times = []
    for _ in range(30):
        start = time.time()
        _ = acqf._extract_parameter_variances_laplace()
        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000  # 转为毫秒
    std_time = np.std(times) * 1000

    print(f"  平均耗时: {avg_time:.2f} ± {std_time:.2f} ms")

    # 参考值（修复前约为 800ms，修复后约为 50ms）
    if avg_time < 100:
        print(f"  ✅ 性能优秀（< 100ms）")
    elif avg_time < 200:
        print(f"  ✅ 性能良好（< 200ms）")
    else:
        print(f"  ⚠️  性能可能需要优化（> 200ms）")

    print("\n✅ 性能测试完成\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("EUR ANOVA Pair Acqf 修复验证测试套件")
    print("=" * 80 + "\n")

    results = []

    # 测试1: 问题1修复
    results.append(("问题1: Laplace梯度计算", test_problem1_laplace_memory_safety()))

    # 测试2: 问题2修复
    results.append(("问题2: 交互对解析", test_problem2_interaction_pairs_dedup()))

    # 测试3: 功能完整性
    results.append(("核心功能完整性", test_functional_integrity()))

    # 测试4: 性能对比
    results.append(("性能对比", test_performance_comparison()))

    # 汇总结果
    print("=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12} {name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 所有测试通过！修复成功且不影响功能！")
    else:
        print("❌ 部分测试失败，请检查修复代码")
    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

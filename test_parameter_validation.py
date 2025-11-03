"""
参数验证测试：验证配置错误能够被及时捕获

测试目标：
1. 正确配置：不应抛出异常
2. tau1 <= tau2：应抛出 ValueError
3. lambda_max < lambda_min：应抛出 ValueError
4. gamma_max < gamma_min：应抛出 ValueError
5. tau_n_max <= tau_n_min：应抛出 ValueError
"""

import torch
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood

from eur_anova_pair_acquisition import EURAnovaPairAcqf


def test_correct_configuration():
    """测试：正确配置不应抛出异常"""
    print("=" * 80)
    print("测试1: 正确配置（不应抛出异常）")
    print("=" * 80)

    X_train = torch.randn(20, 3, dtype=torch.float64)
    y_train = torch.randn(20, 1, dtype=torch.float64)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())

    try:
        acqf = EURAnovaPairAcqf(
            model=model,
            interaction_pairs=[(0, 1)],
            tau1=0.7,
            tau2=0.3,
            lambda_min=0.1,
            lambda_max=1.0,
            gamma_max=0.5,
            gamma_min=0.1,
            tau_n_min=3,
            tau_n_max=40,
        )
        print("  ✅ 正确配置成功创建")
        print(f"     tau1={acqf.tau1}, tau2={acqf.tau2}")
        print(f"     lambda_min={acqf.lambda_min}, lambda_max={acqf.lambda_max}")
        print(f"     gamma_min={acqf.gamma_min}, gamma_max={acqf.gamma_max}")
        print(f"     tau_n_min={acqf.tau_n_min}, tau_n_max={acqf.tau_n_max}")
        return True
    except Exception as e:
        print(f"  ❌ 正确配置却抛出异常: {e}")
        return False


def test_tau_ordering_error():
    """测试：tau1 <= tau2 应抛出 ValueError"""
    print("\n" + "=" * 80)
    print("测试2: tau1 <= tau2（应抛出 ValueError）")
    print("=" * 80)

    X_train = torch.randn(20, 3, dtype=torch.float64)
    y_train = torch.randn(20, 1, dtype=torch.float64)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())

    test_cases = [
        {"tau1": 0.3, "tau2": 0.7, "name": "tau1 < tau2"},
        {"tau1": 0.5, "tau2": 0.5, "name": "tau1 = tau2"},
    ]

    all_passed = True

    for tc in test_cases:
        print(f"\n  【子测试】{tc['name']}")
        try:
            acqf = EURAnovaPairAcqf(
                model=model,
                interaction_pairs=[(0, 1)],
                tau1=tc["tau1"],
                tau2=tc["tau2"],
            )
            print(f"    ❌ 未抛出异常（tau1={tc['tau1']}, tau2={tc['tau2']}）")
            all_passed = False
        except ValueError as e:
            if "tau1 must be > tau2" in str(e):
                print(f"    ✅ 正确捕获错误:")
                print(f"       {e}")
            else:
                print(f"    ❌ 错误信息不正确: {e}")
                all_passed = False
        except Exception as e:
            print(f"    ❌ 抛出了错误类型的异常: {type(e).__name__}: {e}")
            all_passed = False

    return all_passed


def test_lambda_range_error():
    """测试：lambda_max < lambda_min 应抛出 ValueError"""
    print("\n" + "=" * 80)
    print("测试3: lambda_max < lambda_min（应抛出 ValueError）")
    print("=" * 80)

    X_train = torch.randn(20, 3, dtype=torch.float64)
    y_train = torch.randn(20, 1, dtype=torch.float64)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())

    print(f"\n  【测试】lambda_max=0.1 < lambda_min=1.0")
    try:
        acqf = EURAnovaPairAcqf(
            model=model, interaction_pairs=[(0, 1)], lambda_min=1.0, lambda_max=0.1
        )
        print(f"    ❌ 未抛出异常")
        return False
    except ValueError as e:
        if "lambda_max must be >= lambda_min" in str(e):
            print(f"    ✅ 正确捕获错误:")
            print(f"       {e}")
            return True
        else:
            print(f"    ❌ 错误信息不正确: {e}")
            return False
    except Exception as e:
        print(f"    ❌ 抛出了错误类型的异常: {type(e).__name__}: {e}")
        return False


def test_gamma_range_error():
    """测试：gamma_max < gamma_min 应抛出 ValueError"""
    print("\n" + "=" * 80)
    print("测试4: gamma_max < gamma_min（应抛出 ValueError）")
    print("=" * 80)

    X_train = torch.randn(20, 3, dtype=torch.float64)
    y_train = torch.randn(20, 1, dtype=torch.float64)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())

    print(f"\n  【测试】gamma_max=0.1 < gamma_min=0.5")
    try:
        acqf = EURAnovaPairAcqf(
            model=model, interaction_pairs=[(0, 1)], gamma_min=0.5, gamma_max=0.1
        )
        print(f"    ❌ 未抛出异常")
        return False
    except ValueError as e:
        if "gamma_max must be >= gamma_min" in str(e):
            print(f"    ✅ 正确捕获错误:")
            print(f"       {e}")
            return True
        else:
            print(f"    ❌ 错误信息不正确: {e}")
            return False
    except Exception as e:
        print(f"    ❌ 抛出了错误类型的异常: {type(e).__name__}: {e}")
        return False


def test_tau_n_ordering_error():
    """测试：tau_n_max <= tau_n_min 应抛出 ValueError"""
    print("\n" + "=" * 80)
    print("测试5: tau_n_max <= tau_n_min（应抛出 ValueError）")
    print("=" * 80)

    X_train = torch.randn(20, 3, dtype=torch.float64)
    y_train = torch.randn(20, 1, dtype=torch.float64)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())

    test_cases = [
        {"tau_n_min": 40, "tau_n_max": 3, "name": "tau_n_max < tau_n_min"},
        {"tau_n_min": 10, "tau_n_max": 10, "name": "tau_n_max = tau_n_min"},
    ]

    all_passed = True

    for tc in test_cases:
        print(f"\n  【子测试】{tc['name']}")
        try:
            acqf = EURAnovaPairAcqf(
                model=model,
                interaction_pairs=[(0, 1)],
                tau_n_min=tc["tau_n_min"],
                tau_n_max=tc["tau_n_max"],
            )
            print(
                f"    ❌ 未抛出异常（tau_n_min={tc['tau_n_min']}, tau_n_max={tc['tau_n_max']}）"
            )
            all_passed = False
        except ValueError as e:
            if "tau_n_max must be > tau_n_min" in str(e):
                print(f"    ✅ 正确捕获错误:")
                print(f"       {e}")
            else:
                print(f"    ❌ 错误信息不正确: {e}")
                all_passed = False
        except Exception as e:
            print(f"    ❌ 抛出了错误类型的异常: {type(e).__name__}: {e}")
            all_passed = False

    return all_passed


def test_default_values_are_valid():
    """测试：默认值应该是有效的"""
    print("\n" + "=" * 80)
    print("测试6: 默认值应该有效")
    print("=" * 80)

    X_train = torch.randn(20, 3, dtype=torch.float64)
    y_train = torch.randn(20, 1, dtype=torch.float64)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())

    try:
        # 使用所有默认值
        acqf = EURAnovaPairAcqf(model=model, interaction_pairs=[(0, 1)])
        print(f"  ✅ 默认值配置成功:")
        print(f"     tau1={acqf.tau1}, tau2={acqf.tau2}")
        print(f"     lambda_min={acqf.lambda_min}, lambda_max={acqf.lambda_max}")
        print(f"     gamma_min={acqf.gamma_min}, gamma_max={acqf.gamma_max}")
        print(f"     tau_n_min={acqf.tau_n_min}, tau_n_max={acqf.tau_n_max}")
        print(f"     main_weight={acqf.main_weight}")

        # 验证约束
        checks = [
            (acqf.tau1 > acqf.tau2, "tau1 > tau2"),
            (acqf.lambda_max >= acqf.lambda_min, "lambda_max >= lambda_min"),
            (acqf.gamma_max >= acqf.gamma_min, "gamma_max >= gamma_min"),
            (acqf.tau_n_max > acqf.tau_n_min, "tau_n_max > tau_n_min"),
            (acqf.main_weight > 0, "main_weight > 0"),
        ]

        all_valid = True
        for check, name in checks:
            if check:
                print(f"     ✅ {name}")
            else:
                print(f"     ❌ {name} 违反")
                all_valid = False

        return all_valid
    except Exception as e:
        print(f"  ❌ 默认值配置失败: {e}")
        return False


def main():
    """运行所有参数验证测试"""
    print("\n" + "=" * 80)
    print("参数验证测试套件")
    print("=" * 80 + "\n")

    results = []

    results.append(("正确配置", test_correct_configuration()))
    results.append(("tau 顺序验证", test_tau_ordering_error()))
    results.append(("lambda 范围验证", test_lambda_range_error()))
    results.append(("gamma 范围验证", test_gamma_range_error()))
    results.append(("tau_n 顺序验证", test_tau_n_ordering_error()))
    results.append(("默认值有效性", test_default_values_are_valid()))

    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12} {name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 所有参数验证测试通过！配置安全性得到保障！")
    else:
        print("❌ 部分测试失败，请检查参数验证逻辑")
    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

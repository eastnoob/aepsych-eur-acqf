"""
完整测试脚本 - 验证所有功�?
使用 pixi 运行: pixi run python complete_test.py
"""

import sys
import numpy as np
from pathlib import Path
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 直接导入模块
from acquisition_function import VarianceReductionWithCoverageAcqf
from gower_distance import gower_distance, compute_coverage
from gp_variance import GPVarianceCalculator


def print_section(title):
    """打印分隔�?""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_imports():
    """测试 1: 模块导入"""
    print_section("测试 1: 模块导入")

    try:
        # 已在文件开头导�?
        print("�?所有模块导入成�?)
        print(f"  - VarianceReductionWithCoverageAcqf: {VarianceReductionWithCoverageAcqf}")
        print(f"  - gower_distance: {gower_distance}")
        print(f"  - GPVarianceCalculator: {GPVarianceCalculator}")
        return True
    except Exception as e:
        print(f"�?导入失败: {e}")
        traceback.print_exc()
        return False


def test_gower_distance():
    """测试 2: Gower 距离计算"""
    print_section("测试 2: Gower 距离计算")

    try:

        # 测试连续变量
        x1 = np.array([0.0, 0.0, 0.0])
        x2 = np.array([1.0, 1.0, 1.0])
        dist = gower_distance(x1, x2)
        print(f"�?连续变量距离: {dist:.4f}")
        assert 0 <= dist <= 1, "距离应在 [0, 1] 范围�?

        # 测试相同�?
        dist_same = gower_distance(x1, x1)
        print(f"�?相同点距�? {dist_same:.4f}")
        assert dist_same < 1e-6, "相同点距离应�?0"

        # 测试分类变量
        x3 = np.array([1.0, 2.0])
        x4 = np.array([1.0, 3.0])
        variable_types = {0: "categorical", 1: "categorical"}
        dist_cat = gower_distance(x3, x4, variable_types)
        print(f"�?分类变量距离: {dist_cat:.4f}")

        print("�?Gower 距离测试通过")
        return True
    except Exception as e:
        print(f"�?测试失败: {e}")
        traceback.print_exc()
        return False


def test_gp_variance():
    """测试 3: GP 方差计算"""
    print_section("测试 3: GP 方差计算")

    try:

        # 生成测试数据
        np.random.seed(42)
        X = np.random.rand(30, 3)
        y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.1 * np.random.randn(30)

        # 创建并拟�?GP
        gp = GPVarianceCalculator()
        gp.fit(X, y)
        print("�?GP 拟合成功")

        # 测试预测
        X_test = np.random.rand(10, 3)
        y_pred = gp.predict(X_test)
        print(f"�?预测 {len(y_pred)} 个点")

        # 测试主效应方�?
        for i in range(3):
            var = gp.get_main_effect_variance(i)
            print(f"  特征 {i} 方差: {var:.6f}")
            assert var > 0, f"方差应为正数"

        # 测试方差减少
        X_new = np.array([[0.5, 0.5, 0.5]])
        main_var_red, inter_var_red = gp.compute_variance_reduction(X_new)
        print(
            f"�?方差减少计算成功: 主效�?{len(main_var_red)}, 交互 {len(inter_var_red)}"
        )

        print("�?GP 方差计算测试通过")
        return True
    except Exception as e:
        print(f"�?测试失败: {e}")
        traceback.print_exc()
        return False


def test_acquisition_function_basic():
    """测试 4: 基本采集函数"""
    print_section("测试 4: 基本采集函数")

    try:

        # 生成数据
        np.random.seed(42)
        X_train = np.random.rand(30, 3)
        y_train = np.random.rand(30)

        # 创建采集函数
        acq_fn = VarianceReductionWithCoverageAcqf()
        print("�?采集函数创建成功")

        # 拟合
        acq_fn.fit(X_train, y_train)
        print(f"�?拟合完成, 样本�? {len(X_train)}")

        # 评估候选点
        X_candidates = np.random.rand(50, 3)
        scores = acq_fn(X_candidates)
        print(f"�?评估 {len(scores)} 个候选点")
        print(f"  分数范围: [{scores.min():.4f}, {scores.max():.4f}]")

        # 选择最佳点
        next_X, indices = acq_fn.select_next(X_candidates, n_select=3)
        print(f"�?选择�?{len(next_X)} 个点")
        print(f"  索引: {indices}")

        # 检查动态参�?
        lambda_t = acq_fn.get_current_lambda()
        r_t = acq_fn.get_variance_reduction_ratio()
        print(f"�?动态参�? λ_t={lambda_t:.4f}, r_t={r_t:.4f}")

        print("�?基本采集函数测试通过")
        return True
    except Exception as e:
        print(f"�?测试失败: {e}")
        traceback.print_exc()
        return False


def test_acquisition_with_interactions():
    """测试 5: 带交互项的采集函�?""
    print_section("测试 5: 带交互项的采集函�?)

    try:

        # 生成包含交互效应的数�?
        np.random.seed(42)
        X_train = np.random.rand(40, 4)
        y_train = (
            X_train[:, 0]
            + X_train[:, 1]
            + 2 * X_train[:, 0] * X_train[:, 1]  # 交互 0-1
            + X_train[:, 2] * X_train[:, 3]
        )  # 交互 2-3

        # 创建带交互项的采集函�?
        acq_fn = VarianceReductionWithCoverageAcqf(
            interaction_terms=[(0, 1), (2, 3)], lambda_min=0.5, lambda_max=3.0
        )
        print(f"�?创建采集函数，交互项: {acq_fn.interaction_terms}")

        # 拟合
        acq_fn.fit(X_train, y_train)
        print(f"�?拟合完成")

        # 评估并获取分数组�?
        X_candidates = np.random.rand(50, 4)
        total, info, cov = acq_fn(X_candidates, return_components=True)

        print(f"�?分数统计:")
        print(f"  总分:   [{total.min():.4f}, {total.max():.4f}]")
        print(f"  信息:   [{info.min():.4f}, {info.max():.4f}]")
        print(f"  覆盖:   [{cov.min():.4f}, {cov.max():.4f}]")

        # 验证总分是分量之�?
        np.testing.assert_array_almost_equal(total, info + cov, decimal=5)
        print("�?分数组成验证通过")

        print("�?交互项测试通过")
        return True
    except Exception as e:
        print(f"�?测试失败: {e}")
        traceback.print_exc()
        return False


def test_config_file():
    """测试 6: 配置文件加载"""
    print_section("测试 6: 配置文件加载")

    try:

        # 使用示例配置文件
        config_path = Path(__file__).parent / "config_example.ini"

        if not config_path.exists():
            print(f"�?配置文件不存�? {config_path}")
            return True  # 不算失败

        # 从配置加�?
        acq_fn = VarianceReductionWithCoverageAcqf(config_ini_path=config_path)
        print(f"�?从配置文件加载成�?)
        print(f"  lambda_min: {acq_fn.lambda_min}")
        print(f"  lambda_max: {acq_fn.lambda_max}")
        print(f"  tau_1: {acq_fn.tau_1}")
        print(f"  tau_2: {acq_fn.tau_2}")
        print(f"  gamma: {acq_fn.gamma}")
        print(f"  交互�? {acq_fn.interaction_terms}")

        # 测试使用
        np.random.seed(42)
        X_train = np.random.rand(30, 4)
        y_train = np.random.rand(30)
        acq_fn.fit(X_train, y_train)

        X_candidates = np.random.rand(50, 4)
        scores = acq_fn(X_candidates)
        print(f"�?使用配置运行成功，评估了 {len(scores)} 个候选点")

        print("�?配置文件测试通过")
        return True
    except Exception as e:
        print(f"�?测试失败: {e}")
        traceback.print_exc()
        return False


def test_mixed_variables():
    """测试 7: 混合变量类型"""
    print_section("测试 7: 混合变量类型")

    try:

        # 生成混合数据
        np.random.seed(42)
        n_samples = 30
        X_train = np.random.rand(n_samples, 4)
        X_train[:, 2] = np.random.randint(0, 3, n_samples)  # 分类
        X_train[:, 3] = np.random.randint(0, 2, n_samples)  # 分类

        y_train = X_train[:, 0] + 2 * X_train[:, 1] + 0.5 * X_train[:, 2]

        # 定义变量类型
        variable_types = {
            0: "continuous",
            1: "continuous",
            2: "categorical",
            3: "categorical",
        }

        # 创建采集函数
        acq_fn = VarianceReductionWithCoverageAcqf(variable_types=variable_types, gamma=0.5)
        print(f"�?创建混合变量采集函数")
        print(f"  变量类型: {variable_types}")

        # 拟合
        acq_fn.fit(X_train, y_train, variable_types=variable_types)
        print(f"�?拟合完成")

        # 生成混合候选点
        X_candidates = np.random.rand(50, 4)
        X_candidates[:, 2] = np.random.randint(0, 3, 50)
        X_candidates[:, 3] = np.random.randint(0, 2, 50)

        scores = acq_fn(X_candidates)
        print(f"�?评估 {len(scores)} 个混合候选点")
        print(f"  分数范围: [{scores.min():.4f}, {scores.max():.4f}]")

        # 选择最佳点
        next_X, indices = acq_fn.select_next(X_candidates, n_select=3)
        print(f"�?选择�?{len(next_X)} 个点")
        for i, x in enumerate(next_X):
            print(
                f"  �?{i+1}: 连续=[{x[0]:.3f}, {x[1]:.3f}], 分类=[{int(x[2])}, {int(x[3])}]"
            )

        print("�?混合变量测试通过")
        return True
    except Exception as e:
        print(f"�?测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """运行所有测�?""
    print("\n" + "=" * 70)
    print("  Dynamic EUR Acquisition Function - 完整测试")
    print("=" * 70)

    results = []

    # 运行所有测�?
    results.append(("模块导入", test_imports()))
    results.append(("Gower 距离", test_gower_distance()))
    results.append(("GP 方差计算", test_gp_variance()))
    results.append(("基本采集函数", test_acquisition_function_basic()))
    results.append(("交互项采集函�?, test_acquisition_with_interactions()))
    results.append(("配置文件加载", test_config_file()))
    results.append(("混合变量类型", test_mixed_variables()))

    # 打印总结
    print_section("测试总结")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "�?通过" if result else "�?失败"
        print(f"{status:8s} - {name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！功能正常工作！")
        return 0
    else:
        print(f"\n�?{total - passed} 个测试失�?)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

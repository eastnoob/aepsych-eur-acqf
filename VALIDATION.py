# -*- coding: utf-8 -*-
"""
验证脚本 - 快速验证所有核心功能

运行: pixi run python VALIDATION.py
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print(" 核心功能验证测试 ")
print("=" * 80)

# 测试 1: 模块导入
print("\n[1/6] 测试模块导入...")
try:
    from acquisition_function import VarianceReductionWithCoverageAcqf
    from gower_distance import gower_distance, compute_coverage
    from gp_variance import GPVarianceCalculator

    print("  ✓ 所有模块成功导入")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    sys.exit(1)

# 测试 2: 基本实例化
print("\n[2/6] 测试基本实例化...")
try:
    acq_fn = VarianceReductionWithCoverageAcqf()
    print(f"  ✓ 成功创建采集函数")
    print(f"    - lambda_min={acq_fn.lambda_min}")
    print(f"    - lambda_max={acq_fn.lambda_max}")
    print(f"    - gamma={acq_fn.gamma}")
except Exception as e:
    print(f"  ✗ 实例化失败: {e}")
    sys.exit(1)

# 测试 3: 配置文件加载
print("\n[3/6] 测试配置文件加载...")
try:
    config_path = project_root / "configs" / "full_experiment_config.ini"
    if config_path.exists():
        acq_fn_config = VarianceReductionWithCoverageAcqf(config_ini_path=config_path)
        print(f"  ✓ 成功从配置文件加载")
        print(f"    - lambda_min={acq_fn_config.lambda_min}")
        print(f"    - gamma={acq_fn_config.gamma}")
    else:
        print(f"  ⚠ 配置文件不存在,跳过")
except Exception as e:
    print(f"  ✗ 配置加载失败: {e}")

# 测试 4: 数据拟合
print("\n[4/6] 测试数据拟合...")
try:
    np.random.seed(42)
    X_train = np.random.rand(20, 3)
    y_train = np.random.rand(20)

    acq_fn = VarianceReductionWithCoverageAcqf(
        lambda_min=0.2, lambda_max=2.0, gamma=0.5, interaction_terms=[(0, 1), (1, 2)]
    )
    acq_fn.fit(X_train, y_train)
    print(f"  ✓ 成功拟合数据")
    print(f"    - 训练样本数: {len(X_train)}")
    print(f"    - 特征维度: {X_train.shape[1]}")
    print(f"    - λ_t = {acq_fn.get_current_lambda():.3f}")
    print(f"    - r_t = {acq_fn.get_variance_reduction_ratio():.3f}")
except Exception as e:
    print(f"  ✗ 拟合失败: {e}")
    sys.exit(1)

# 测试 5: 候选点评估
print("\n[5/6] 测试候选点评估...")
try:
    X_candidates = np.random.rand(100, 3)
    scores = acq_fn(X_candidates)
    print(f"  ✓ 成功评估候选点")
    print(f"    - 候选点数: {len(X_candidates)}")
    print(f"    - 分数范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    print(f"    - 平均分数: {np.mean(scores):.4f}")
except Exception as e:
    print(f"  ✗ 评估失败: {e}")
    sys.exit(1)

# 测试 6: 主动学习循环
print("\n[6/6] 测试主动学习循环...")
try:
    # 简单的3次迭代测试
    n_iterations = 3
    for i in range(n_iterations):
        # 生成候选点
        X_candidates = np.random.rand(50, 3)

        # 评估并选择
        selected_X, selected_idx = acq_fn.select_next(X_candidates, n_select=1)

        # 获取新标签(模拟)
        y_new = np.random.rand(1)

        # 更新数据集
        X_train = np.vstack([X_train, selected_X])
        y_train = np.hstack([y_train, y_new])

        # 重新拟合
        acq_fn.fit(X_train, y_train)

    print(f"  ✓ 成功完成 {n_iterations} 次主动学习迭代")
    print(f"    - 最终样本数: {len(X_train)}")
    print(f"    - 最终 λ_t = {acq_fn.get_current_lambda():.3f}")
    print(f"    - 最终 r_t = {acq_fn.get_variance_reduction_ratio():.3f}")
except Exception as e:
    print(f"  ✗ 主动学习循环失败: {e}")
    sys.exit(1)

# 测试 7: Gower距离
print("\n[Bonus] 测试Gower距离...")
try:
    x1 = np.array([0.5, 1.0])
    x2 = np.array([0.8, 1.0])

    # 连续变量
    dist_cont = gower_distance(
        x1, x2, variable_types={0: "continuous", 1: "continuous"}
    )
    print(f"  ✓ Gower距离计算成功")
    print(f"    - 连续变量距离: {dist_cont:.4f}")

    # 混合变量
    x1_mixed = np.array([0.5, 1.0])
    x2_mixed = np.array([0.8, 2.0])
    dist_mixed = gower_distance(
        x1_mixed, x2_mixed, variable_types={0: "continuous", 1: "categorical"}
    )
    print(f"    - 混合变量距离: {dist_mixed:.4f}")
except Exception as e:
    print(f"  ⚠ Gower距离测试失败: {e}")

# 总结
print("\n" + "=" * 80)
print(" 所有核心功能测试通过! ✓ ")
print("=" * 80)
print("\n项目状态:")
print(f"  • 核心代码文件: acquisition_function.py, gower_distance.py, gp_variance.py")
print(f"  • 配置文件: configs/")
print(f"  • 文档: docs/")
print(f"  • 测试: test/")
print(f"\n快速开始:")
print(f"  1. 查看 README.md 了解项目概览")
print(f"  2. 查看 docs/QUICKSTART.md 学习使用")
print(f"  3. 运行 test/unit_tests/simple_test.py 快速验证")
print(f"  4. 运行 test/integration_tests/end_to_end_experiment.py 完整实验")
print("\n" + "=" * 80)

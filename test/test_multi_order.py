"""
单元测试：多阶交互ANOVA采集函数

测试覆盖：
1. 只启用主效应
2. 主效应 + 二阶交互
3. 主效应 + 二阶 + 三阶交互
4. 配置解析
5. 动态权重
6. 模块独立性
"""

import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood


def create_mock_model(n_dims=4, n_train=10):
    """创建模拟GP模型"""
    torch.manual_seed(42)
    np.random.seed(42)

    X_train = torch.rand(n_train, n_dims, dtype=torch.float64)
    y_train = torch.randn(n_train, 1, dtype=torch.float64)

    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())
    model.eval()

    return model


def test_main_only():
    """测试：只启用主效应"""
    print("\n" + "=" * 70)
    print("测试 1: 只启用主效应")
    print("=" * 70)

    from eur_anova_multi import EURAnovaMultiAcqf

    model = create_mock_model(n_dims=4, n_train=10)

    acqf = EURAnovaMultiAcqf(
        model,
        enable_main=True,
        enable_pairwise=False,  # 关闭二阶
        enable_threeway=False,  # 关闭三阶
        debug_components=True,
    )

    # 评估候选点
    X_candidates = torch.rand(5, 4, dtype=torch.float64)
    # 在 BoTorch 中，acqf 应使用 (B, q, d) 的输入，该处 q=1
    scores = acqf(X_candidates.unsqueeze(1))

    print(f"候选点数量: {X_candidates.shape[0]}")
    print(f"得分范围: [{scores.min():.4f}, {scores.max():.4f}]")

    # 检查诊断信息
    diag = acqf.get_diagnostics()
    print(f"启用主效应: {diag['enable_main']}")
    print(f"启用二阶: {diag['enable_pairwise']}")
    print(f"启用三阶: {diag['enable_threeway']}")
    print(f"二阶交互数: {diag['n_pairs']}")
    print(f"三阶交互数: {diag['n_triplets']}")

    assert diag["enable_main"] == True
    assert diag["enable_pairwise"] == False
    assert diag["n_pairs"] == 0
    assert diag["n_triplets"] == 0

    print("✅ 测试通过！")


def test_main_plus_pairwise():
    """测试：主效应 + 二阶交互"""
    print("\n" + "=" * 70)
    print("测试 2: 主效应 + 二阶交互")
    print("=" * 70)

    from eur_anova_multi import EURAnovaMultiAcqf

    model = create_mock_model(n_dims=4, n_train=10)

    acqf = EURAnovaMultiAcqf(
        model,
        enable_main=True,
        interaction_pairs=[(0, 1), (2, 3)],  # 两个二阶交互
        enable_threeway=False,  # 关闭三阶
        lambda_2=1.0,  # 固定二阶权重
        debug_components=True,
    )

    X_candidates = torch.rand(5, 4, dtype=torch.float64)
    scores = acqf(X_candidates.unsqueeze(1))

    print(f"候选点数量: {X_candidates.shape[0]}")
    print(f"得分范围: [{scores.min():.4f}, {scores.max():.4f}]")

    diag = acqf.get_diagnostics()
    print(f"二阶交互数: {diag['n_pairs']}")
    print(f"二阶交互: {diag['pairs']}")
    print(f"λ_2 权重: {diag['lambda_2']:.4f}")

    assert diag["n_pairs"] == 2
    assert (0, 1) in diag["pairs"]
    assert (2, 3) in diag["pairs"]

    print("✅ 测试通过！")


def test_full_three_way():
    """测试：主效应 + 二阶 + 三阶交互"""
    print("\n" + "=" * 70)
    print("测试 3: 主效应 + 二阶 + 三阶交互")
    print("=" * 70)

    from eur_anova_multi import EURAnovaMultiAcqf

    model = create_mock_model(n_dims=4, n_train=10)

    acqf = EURAnovaMultiAcqf(
        model,
        enable_main=True,
        interaction_pairs=[(0, 1), (1, 2), (2, 3)],
        interaction_triplets=[(0, 1, 2)],  # 一个三阶交互
        lambda_2=1.0,
        lambda_3=0.5,  # 三阶权重
        debug_components=True,
    )

    X_candidates = torch.rand(5, 4, dtype=torch.float64)
    scores = acqf(X_candidates.unsqueeze(1))

    print(f"候选点数量: {X_candidates.shape[0]}")
    print(f"得分范围: [{scores.min():.4f}, {scores.max():.4f}]")

    diag = acqf.get_diagnostics()
    print(f"二阶交互数: {diag['n_pairs']}")
    print(f"三阶交互数: {diag['n_triplets']}")
    print(f"三阶交互: {diag['triplets']}")
    print(f"λ_2 权重: {diag['lambda_2']:.4f}")
    print(f"λ_3 权重: {diag['lambda_3']:.4f}")

    assert diag["n_triplets"] == 1
    assert (0, 1, 2) in diag["triplets"]

    # 打印完整诊断
    acqf.print_diagnostics()

    print("✅ 测试通过！")


def test_config_parsing():
    """测试：配置解析"""
    print("\n" + "=" * 70)
    print("测试 4: 配置解析")
    print("=" * 70)

    from modules.config_parser import (
        parse_interaction_pairs,
        parse_interaction_triplets,
        parse_variable_types,
    )

    # 测试交互对解析
    pairs = parse_interaction_pairs("0,1; 2,3")
    print(f"解析二阶交互: {pairs}")
    assert pairs == [(0, 1), (2, 3)]

    # 测试三元组解析
    triplets = parse_interaction_triplets([(2, 1, 0), (1, 2, 3)])
    print(f"解析三阶交互: {triplets}")
    assert (0, 1, 2) in triplets  # 已规范化
    assert (1, 2, 3) in triplets

    # 测试变量类型解析
    vt = parse_variable_types("categorical, continuous, integer")
    print(f"解析变量类型: {vt}")
    assert vt[0] == "categorical"
    assert vt[1] == "continuous"
    assert vt[2] == "integer"

    print("✅ 测试通过！")


def test_anova_engine_independently():
    """测试：ANOVA引擎独立性"""
    print("\n" + "=" * 70)
    print("测试 5: ANOVA引擎独立性")
    print("=" * 70)

    from modules.anova_effects import (
        ANOVAEffectEngine,
        MainEffect,
        PairwiseEffect,
        ThreeWayEffect,
    )

    # 模拟信息度量函数
    def mock_metric(X):
        return torch.randn(X.shape[0])

    # 模拟局部采样器
    def mock_sampler(X, dims):
        B = X.shape[0]
        local_num = 4
        return X.unsqueeze(1).repeat(1, local_num, 1).reshape(B * local_num, X.shape[1])

    engine = ANOVAEffectEngine(metric_fn=mock_metric, local_sampler=mock_sampler)

    # 定义效应
    effects = [
        MainEffect(0),
        MainEffect(1),
        PairwiseEffect(0, 1),
        ThreeWayEffect(0, 1, 2),
    ]

    X_test = torch.rand(5, 4)
    results = engine.compute_effects(X_test, effects)

    print(f"基线形状: {results['baseline'].shape}")
    print(f"原始结果数量: {len(results['raw_results'])}")
    print(f"贡献数量: {len(results['contributions'])}")
    print(f"聚合阶数: {list(results['aggregated'].keys())}")

    assert "order_1" in results["aggregated"]
    assert "order_2" in results["aggregated"]
    assert "order_3" in results["aggregated"]

    print("✅ 测试通过！")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#" * 70)
    print("# 多阶交互ANOVA采集函数 - 单元测试套件")
    print("#" * 70)

    tests = [
        test_main_only,
        test_main_plus_pairwise,
        test_full_three_way,
        test_config_parsing,
        test_anova_engine_independently,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ 测试失败: {test_func.__name__}")
            print(f"错误: {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    print("\n" + "#" * 70)
    print(f"# 测试结果: {passed} 通过, {failed} 失败")
    print("#" * 70)


if __name__ == "__main__":
    run_all_tests()

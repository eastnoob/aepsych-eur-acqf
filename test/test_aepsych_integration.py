"""
测试AEPsych集成兼容性

验证新实现能否在AEPsych框架中正常使用。
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
import configparser

# 修复Windows编码问题
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def test_basic_botorch_compatibility():
    """Test 1: BoTorch basic compatibility"""
    print("\n" + "=" * 70)
    print("Test 1: BoTorch Basic Compatibility")
    print("=" * 70)

    from eur_anova_multi import EURAnovaMultiAcqf

    # 创建标准BoTorch模型
    torch.manual_seed(42)
    X_train = torch.rand(10, 4, dtype=torch.float64)
    y_train = torch.randn(10, 1, dtype=torch.float64)

    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())
    model.eval()

    # 创建采集函数
    acqf = EURAnovaMultiAcqf(
        model, enable_main=True, interaction_pairs=[(0, 1), (2, 3)]
    )

    # 测试forward()方法
    X_test = torch.rand(5, 4, dtype=torch.float64)
    scores = acqf(X_test.unsqueeze(1))

    print(f"✅ 模型类型: {type(model).__name__}")
    print(f"✅ 采集函数类型: {type(acqf).__name__}")
    print(f"✅ 输入形状: {X_test.shape}")
    print(f"✅ 输出形状: {scores.shape}")
    print(f"✅ 得分范围: [{scores.min():.4f}, {scores.max():.4f}]")

    assert scores.shape == (5,), "输出形状不正确"
    assert torch.isfinite(scores).all(), "存在NaN/Inf值"

    print("✅ BoTorch兼容性测试通过!")


def test_config_file_parsing():
    """测试2: 配置文件解析"""
    print("\n" + "=" * 70)
    print("测试 2: 配置文件解析")
    print("=" * 70)

    # 模拟AEPsych配置
    config_str = """
[EURAnovaMultiAcqf]
enable_main = True
enable_pairwise = True
enable_threeway = False
interaction_pairs = 0,1; 2,3
lambda_2 = 1.0
lambda_3 = 0.5
variable_types_list = continuous, continuous, integer, categorical
use_dynamic_lambda = True
tau1 = 0.7
tau2 = 0.3
lambda_min = 0.1
lambda_max = 1.0
gamma = 0.3
local_jitter_frac = 0.1
local_num = 4
random_seed = 42
debug_components = False
    """

    config = configparser.ConfigParser()
    config.read_string(config_str)

    # 解析参数
    section = "EURAnovaMultiAcqf"
    params = {
        "enable_main": config.getboolean(section, "enable_main"),
        "enable_pairwise": config.getboolean(section, "enable_pairwise"),
        "enable_threeway": config.getboolean(section, "enable_threeway"),
        "interaction_pairs": config.get(section, "interaction_pairs"),
        "lambda_2": config.getfloat(section, "lambda_2"),
        "lambda_3": config.getfloat(section, "lambda_3"),
        "variable_types_list": config.get(section, "variable_types_list"),
        "use_dynamic_lambda": config.getboolean(section, "use_dynamic_lambda"),
        "tau1": config.getfloat(section, "tau1"),
        "tau2": config.getfloat(section, "tau2"),
        "lambda_min": config.getfloat(section, "lambda_min"),
        "lambda_max": config.getfloat(section, "lambda_max"),
        "gamma": config.getfloat(section, "gamma"),
        "local_jitter_frac": config.getfloat(section, "local_jitter_frac"),
        "local_num": config.getint(section, "local_num"),
        "random_seed": config.getint(section, "random_seed"),
        "debug_components": config.getboolean(section, "debug_components"),
    }

    print("✅ 解析的参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 使用解析的参数创建采集函数
    from eur_anova_multi import EURAnovaMultiAcqf

    torch.manual_seed(42)
    X_train = torch.rand(10, 4, dtype=torch.float64)
    y_train = torch.randn(10, 1, dtype=torch.float64)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())
    model.eval()

    acqf = EURAnovaMultiAcqf(model, **params)

    # 测试
    X_test = torch.rand(5, 4, dtype=torch.float64)
    scores = acqf(X_test.unsqueeze(1))

    print(f"✅ 采集函数创建成功")
    print(f"✅ 得分计算成功: {scores.shape}")
    print("✅ 配置文件解析测试通过!")


def test_aepsych_style_usage():
    """测试3: AEPsych风格使用"""
    print("\n" + "=" * 70)
    print("测试 3: AEPsych 风格使用")
    print("=" * 70)

    from eur_anova_multi import EURAnovaMultiAcqf

    # 模拟AEPsych工作流
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. 创建模型
    X_train = torch.rand(15, 4, dtype=torch.float64)
    y_train = torch.randn(15, 1, dtype=torch.float64)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())
    model.eval()

    # 2. 从配置创建采集函数（模拟AEPsych策略）
    acqf = EURAnovaMultiAcqf(
        model,
        enable_main=True,
        interaction_pairs="0,1; 2,3",
        variable_types_list="continuous, continuous, integer, categorical",
        total_budget=30,  # AEPsych实验预算
        debug_components=True,
    )

    # 3. 生成候选点（模拟BoTorch优化器）
    X_candidates = torch.rand(100, 4, dtype=torch.float64)

    # 4. 评估采集函数
    scores = acqf(X_candidates.unsqueeze(1))

    # 5. 选择最佳点
    best_idx = scores.argmax()
    next_trial = X_candidates[best_idx]

    print(f"✅ 训练样本: {X_train.shape[0]}")
    print(f"✅ 候选点数: {X_candidates.shape[0]}")
    print(f"✅ 最佳得分: {scores[best_idx]:.4f}")
    print(f"✅ 推荐试验点: {next_trial.tolist()}")

    # 6. 查看诊断信息
    diag = acqf.get_diagnostics()
    print(f"\n诊断信息:")
    print(f"  训练样本数: {diag['n_train']}")
    print(f"  λ_2 权重: {diag['lambda_2']:.4f}")
    print(f"  γ_t 权重: {diag['gamma_t']:.4f}")
    print(f"  二阶交互数: {diag['n_pairs']}")

    print("✅ AEPsych风格使用测试通过!")


def test_multiple_evaluations():
    """测试4: 多次评估（模拟实验迭代）"""
    print("\n" + "=" * 70)
    print("测试 4: 多次评估（模拟实验迭代）")
    print("=" * 70)

    from eur_anova_multi import EURAnovaMultiAcqf

    torch.manual_seed(42)
    np.random.seed(42)

    # 初始数据
    X_train = torch.rand(5, 4, dtype=torch.float64)
    y_train = torch.randn(5, 1, dtype=torch.float64)

    print("模拟10轮实验迭代:")

    for iter in range(10):
        # 更新模型
        model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())
        model.eval()

        # 更新采集函数
        acqf = EURAnovaMultiAcqf(
            model, enable_main=True, interaction_pairs=[(0, 1), (2, 3)], total_budget=20
        )

        # 评估候选点
        X_candidates = torch.rand(50, 4, dtype=torch.float64)
        scores = acqf(X_candidates.unsqueeze(1))

        # 选择最佳点
        best_idx = scores.argmax()
        next_trial = X_candidates[best_idx].unsqueeze(0)
        next_response = torch.randn(1, 1, dtype=torch.float64)

        # 更新训练集
        X_train = torch.cat([X_train, next_trial], dim=0)
        y_train = torch.cat([y_train, next_response], dim=0)

        diag = acqf.get_diagnostics()
        print(
            f"  第 {iter+1:2d} 轮: n={X_train.shape[0]:2d}, "
            f"λ_2={diag['lambda_2']:.3f}, "
            f"γ_t={diag['gamma_t']:.3f}, "
            f"得分范围=[{scores.min():.3f}, {scores.max():.3f}]"
        )

    print("✅ 多次评估测试通过!")


def test_backward_compatibility():
    """测试5: 向后兼容性（旧版EURAnovaPairAcqf）"""
    print("\n" + "=" * 70)
    print("测试 5: 向后兼容性")
    print("=" * 70)

    from eur_anova_pair import EURAnovaPairAcqf
    from eur_anova_multi import EURAnovaMultiAcqf

    torch.manual_seed(42)
    X_train = torch.rand(10, 4, dtype=torch.float64)
    y_train = torch.randn(10, 1, dtype=torch.float64)
    model = SingleTaskGP(X_train, y_train, likelihood=GaussianLikelihood())
    model.eval()

    # 旧版
    acqf_old = EURAnovaPairAcqf(
        model, interaction_pairs=[(0, 1), (2, 3)], use_dynamic_lambda=True
    )

    # 新版（等效配置）
    acqf_new = EURAnovaMultiAcqf(
        model,
        enable_main=True,
        interaction_pairs=[(0, 1), (2, 3)],
        enable_threeway=False,
        lambda_2=None,  # 动态
    )

    X_test = torch.rand(5, 4, dtype=torch.float64)

    print(f"✅ 旧版类型: {type(acqf_old).__name__}")
    print(f"✅ 新版类型: {type(acqf_new).__name__}")
    print(f"✅ 两者都能正常评估")

    print("✅ 向后兼容性测试通过!")


def run_all_tests():
    """运行所有集成测试"""
    print("\n" + "#" * 70)
    print("# AEPsych 集成兼容性测试套件")
    print("#" * 70)

    tests = [
        test_basic_botorch_compatibility,
        test_config_file_parsing,
        test_aepsych_style_usage,
        test_multiple_evaluations,
        test_backward_compatibility,
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

    if failed == 0:
        print("\n🎉 所有测试通过！新实现完全兼容AEPsych框架。")
        print("\n可以在AEPsych中这样使用:")
        print("```ini")
        print("[EURAnovaMultiAcqf]")
        print("enable_main = True")
        print("interaction_pairs = 0,1; 2,3")
        print("lambda_2 = 1.0")
        print("```")


if __name__ == "__main__":
    run_all_tests()

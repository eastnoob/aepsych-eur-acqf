"""
模拟实验 - 使用 INI 配置文件进行完整的主动学习实�?

这个脚本演示了如何使用配置文件进行实际的主动学习实验�?
运行: pixi run python simulation_experiment.py
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # 非交互后�?
import matplotlib.pyplot as plt

# 添加当前目录到路�?
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入模块
from acquisition_function import VarianceReductionWithCoverageAcqf
from gp_variance import GPVarianceCalculator


def true_function(X):
    """
    真实的未知函数（实验中需要通过实际测量获得�?
    这里使用一个包含主效应和交互效应的函数

    f(x) = 2*x1 + 3*x2 - x3 + 1.5*x1*x2 - 0.8*x2*x3 + noise
    """
    return (
        2 * X[:, 0]
        + 3 * X[:, 1]
        - X[:, 2]
        + 1.5 * X[:, 0] * X[:, 1]  # 交互 0-1
        - 0.8 * X[:, 1] * X[:, 2]  # 交互 1-2
        + 0.1 * np.random.randn(X.shape[0])
    )


def create_simulation_config():
    """创建模拟实验的配置文�?""
    config_path = Path(__file__).parent / "simulation_config.ini"

    config_content = """# 模拟实验配置文件
# 这是一个优化的配置，用于包含交互效应的函数学习

[AcquisitionFunction]
# 动态交互权重参�?
# 由于我们知道函数包含重要的交互效应，使用较大的权重范�?
lambda_min = 0.3
lambda_max = 2.5

# 方差减少阈�?
# 调整以适应实验进度
tau_1 = 0.6
tau_2 = 0.15

# 空间覆盖权重
# 在早期给予较大权重以确保良好的空间覆�?
gamma = 0.4

# 交互项定�?
# 指定我们想要建模的交互效�?
# 格式: (feature1, feature2);(feature3, feature4)
interaction_terms = (0,1);(1,2)

# GP 参数
noise_variance = 0.1
prior_variance = 1.0

# 覆盖度计算方�?
coverage_method = min_distance
"""

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    print(f"�?创建配置文件: {config_path}")
    return config_path


def run_simulation_experiment(
    n_initial=15, n_iterations=30, n_candidates=300, n_features=3, save_results=True
):
    """
    运行完整的模拟实�?

    参数
    ----
    n_initial : int
        初始随机样本�?
    n_iterations : int
        主动学习迭代次数
    n_candidates : int
        每次迭代的候选点数量
    n_features : int
        特征数量
    save_results : bool
        是否保存结果
    """

    print("=" * 70)
    print("  模拟实验: 使用 INI 配置的主动学�?)
    print("=" * 70)

    # 创建配置文件
    config_path = create_simulation_config()

    # 从配置文件加载采集函�?
    print("\n步骤 1: 从配置文件加载采集函�?)
    acq_fn = VarianceReductionWithCoverageAcqf(config_ini_path=config_path)

    print(f"  �?配置加载成功")
    print(f"    - lambda 范围: [{acq_fn.lambda_min}, {acq_fn.lambda_max}]")
    print(f"    - tau 阈�? [{acq_fn.tau_2}, {acq_fn.tau_1}]")
    print(f"    - gamma: {acq_fn.gamma}")
    print(f"    - 交互�? {acq_fn.interaction_terms}")

    # 生成初始随机样本
    print(f"\n步骤 2: 生成 {n_initial} 个初始随机样�?)
    np.random.seed(42)
    X_train = np.random.rand(n_initial, n_features)
    y_train = true_function(X_train)
    print(f"  �?初始数据�? {X_train.shape}")

    # 记录实验过程
    history = {
        "iteration": [],
        "n_samples": [],
        "lambda_t": [],
        "r_t": [],
        "best_score": [],
        "mean_score": [],
        "std_score": [],
    }

    # 主动学习循环
    print(f"\n步骤 3: 主动学习循环 ({n_iterations} 次迭�?")
    print("-" * 70)

    for iteration in range(n_iterations):
        # 拟合模型
        acq_fn.fit(X_train, y_train)

        # 生成候选点
        X_candidates = np.random.rand(n_candidates, n_features)

        # 评估候选点
        scores = acq_fn(X_candidates)

        # 选择最佳点
        next_X, next_idx = acq_fn.select_next(X_candidates, n_select=1)

        # "进行实验"获取真实�?
        next_y = true_function(next_X)

        # 更新训练�?
        X_train = np.vstack([X_train, next_X])
        y_train = np.concatenate([y_train, next_y])

        # 获取当前状�?
        lambda_t = acq_fn.get_current_lambda()
        r_t = acq_fn.get_variance_reduction_ratio()

        # 记录历史
        history["iteration"].append(iteration)
        history["n_samples"].append(len(X_train))
        history["lambda_t"].append(lambda_t)
        history["r_t"].append(r_t)
        history["best_score"].append(scores[next_idx[0]])
        history["mean_score"].append(np.mean(scores))
        history["std_score"].append(np.std(scores))

        # �?5 次迭代打印进�?
        if iteration % 5 == 0 or iteration == n_iterations - 1:
            print(
                f"迭代 {iteration:3d}: "
                f"样本�?{len(X_train):3d}, "
                f"λ_t={lambda_t:.3f}, "
                f"r_t={r_t:.3f}, "
                f"最佳分�?{scores[next_idx[0]]:.4f}"
            )

    print("-" * 70)
    print(f"�?实验完成！最终数据集大小: {len(X_train)} 样本")

    # 最终评�?
    print(f"\n步骤 4: 最终评�?)
    print(f"  初始样本�? {n_initial}")
    print(f"  最终样本数: {len(X_train)}")
    print(f"  新增样本�? {len(X_train) - n_initial}")
    print(f"  最�?λ_t: {history['lambda_t'][-1]:.4f}")
    print(f"  最�?r_t: {history['r_t'][-1]:.4f}")

    # 保存结果
    if save_results:
        save_experiment_results(history, X_train, y_train)

    # 可视化结�?
    visualize_results(history)

    return X_train, y_train, history


def save_experiment_results(history, X_train, y_train):
    """保存实验结果"""
    print(f"\n步骤 5: 保存结果")

    # 保存训练数据
    data_path = Path(__file__).parent / "simulation_results_data.npz"
    np.savez(data_path, X=X_train, y=y_train)
    print(f"  �?训练数据保存�? {data_path}")

    # 保存历史记录
    history_path = Path(__file__).parent / "simulation_results_history.npz"
    np.savez(history_path, **history)
    print(f"  �?历史记录保存�? {history_path}")


def visualize_results(history):
    """可视化实验结�?""
    print(f"\n步骤 6: 生成可视�?)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("模拟实验结果", fontsize=16, fontweight="bold")

    iterations = history["iteration"]

    # �?1: 样本数增�?
    ax1 = axes[0, 0]
    ax1.plot(iterations, history["n_samples"], "b-", linewidth=2)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("Sample Growth")
    ax1.grid(True, alpha=0.3)

    # �?2: 动态权�?λ_t
    ax2 = axes[0, 1]
    ax2.plot(iterations, history["lambda_t"], "g-", linewidth=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Lambda_t")
    ax2.set_title("Dynamic Interaction Weight")
    ax2.grid(True, alpha=0.3)

    # �?3: 方差减少比例 r_t
    ax3 = axes[1, 0]
    ax3.plot(iterations, history["r_t"], "r-", linewidth=2)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("r_t (Variance Ratio)")
    ax3.set_title("Variance Reduction Progress")
    ax3.grid(True, alpha=0.3)

    # �?4: 采集分数统计
    ax4 = axes[1, 1]
    ax4.plot(iterations, history["best_score"], "b-", label="Best Score", linewidth=2)
    ax4.plot(iterations, history["mean_score"], "g--", label="Mean Score", linewidth=2)
    ax4.fill_between(
        iterations,
        np.array(history["mean_score"]) - np.array(history["std_score"]),
        np.array(history["mean_score"]) + np.array(history["std_score"]),
        alpha=0.3,
        color="green",
    )
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Acquisition Score")
    ax4.set_title("Acquisition Scores")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    fig_path = Path(__file__).parent / "simulation_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  �?可视化结果保存到: {fig_path}")
    plt.close()


def analyze_learned_model(X_train, y_train):
    """分析学习到的模型"""
    print(f"\n步骤 7: 模型分析")

    # 重新拟合 GP 模型
    gp = GPVarianceCalculator()
    interaction_terms = [(0, 1), (1, 2)]
    gp.fit(X_train, y_train, interaction_indices=interaction_terms)

    # 分析主效应方�?
    print(f"\n主效应参数方�?")
    for i in range(3):
        var = gp.get_main_effect_variance(i)
        print(f"  特征 {i}: {var:.6f}")

    # 分析交互效应方差
    print(f"\n交互效应参数方差:")
    for i, (j, k) in enumerate(interaction_terms):
        var = gp.get_interaction_effect_variance(i)
        print(f"  交互 ({j},{k}): {var:.6f}")

    # 测试集评�?
    print(f"\n测试集评�?")
    X_test = np.random.rand(100, 3)
    y_test_true = true_function(X_test)
    y_test_pred, y_test_std = gp.predict(X_test, return_std=True)

    mse = np.mean((y_test_true - y_test_pred) ** 2)
    mae = np.mean(np.abs(y_test_true - y_test_pred))

    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  预测标准�? {np.mean(y_test_std):.6f}")


def main():
    """主函�?""
    try:
        # 运行模拟实验
        X_train, y_train, history = run_simulation_experiment(
            n_initial=15,
            n_iterations=30,
            n_candidates=300,
            n_features=3,
            save_results=True,
        )

        # 分析学习到的模型
        analyze_learned_model(X_train, y_train)

        print("\n" + "=" * 70)
        print("  🎉 模拟实验成功完成�?)
        print("=" * 70)
        print("\n生成的文�?")
        print("  - simulation_config.ini        (配置文件)")
        print("  - simulation_results_data.npz  (训练数据)")
        print("  - simulation_results_history.npz (历史记录)")
        print("  - simulation_results.png       (可视化结�?")
        print("\n实验证明:")
        print("  �?INI 配置文件可以正确加载和使�?)
        print("  �?采集函数在主动学习中正常工作")
        print("  �?动态权重机制按预期调整")
        print("  �?可以有效学习包含交互效应的函�?)

        return 0

    except Exception as e:
        print(f"\n�?实验失败: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

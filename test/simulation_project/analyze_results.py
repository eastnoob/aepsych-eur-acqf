"""
分析实验结果

展示如何加载和分析模拟实验的数据
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_latest_results(results_dir: str = "results"):
    """加载最新的实验结果"""
    # 查找最新的文件
    files = [f for f in os.listdir(results_dir) if f.startswith("experiment_data_")]
    if not files:
        raise FileNotFoundError("未找到实验数据文件")

    latest = sorted(files)[-1]
    timestamp = latest.replace("experiment_data_", "").replace(".csv", "")

    print(f"加载时间戳: {timestamp}")

    # 加载所有相关文件
    data = {
        "trials": pd.read_csv(
            os.path.join(results_dir, f"experiment_data_{timestamp}.csv")
        ),
        "metadata": json.load(
            open(os.path.join(results_dir, f"experiment_metadata_{timestamp}.json"))
        ),
        "training": np.load(
            os.path.join(results_dir, f"training_data_{timestamp}.npz")
        ),
        "evaluation": np.load(
            os.path.join(results_dir, f"evaluation_results_{timestamp}.npz")
        ),
    }

    return data, timestamp


def analyze_adaptive_behavior(df: pd.DataFrame):
    """分析自适应行为"""
    print("\n" + "=" * 80)
    print(" 自适应行为分析")
    print("=" * 80)

    # 按阶段分组统计
    init_data = df[df["phase"] == "initialization"]
    opt_data = df[df["phase"] == "optimization"]

    print(f"\n初始化阶段 (n={len(init_data)}):")
    print(
        f"  - 响应均值: {init_data['response'].mean():.3f} ± {init_data['response'].std():.3f}"
    )
    print(
        f"  - 响应范围: [{init_data['response'].min():.3f}, {init_data['response'].max():.3f}]"
    )

    print(f"\n优化阶段 (n={len(opt_data)}):")
    print(
        f"  - 响应均值: {opt_data['response'].mean():.3f} ± {opt_data['response'].std():.3f}"
    )
    print(
        f"  - 响应范围: [{opt_data['response'].min():.3f}, {opt_data['response'].max():.3f}]"
    )
    print(
        f"  - λ_t 变化: {opt_data['lambda_t'].min():.3f} → {opt_data['lambda_t'].max():.3f}"
    )
    print(f"  - r_t 变化: {opt_data['r_t'].max():.3f} → {opt_data['r_t'].min():.3f}")
    print(
        f"  - 采集得分变化: {opt_data['acq_score'].max():.4f} → {opt_data['acq_score'].min():.4f}"
    )

    # 计算观测误差(true_value - response)
    init_errors = (init_data["true_value"] - init_data["response"]).abs()
    opt_errors = (opt_data["true_value"] - opt_data["response"]).abs()

    print(f"\n观测误差:")
    print(f"  - 初始化阶段: {init_errors.mean():.3f} ± {init_errors.std():.3f}")
    print(f"  - 优化阶段: {opt_errors.mean():.3f} ± {opt_errors.std():.3f}")


def analyze_exploration_exploitation(df: pd.DataFrame):
    """分析探索-开发权衡"""
    print("\n" + "=" * 80)
    print(" 探索-开发权衡分析")
    print("=" * 80)

    opt_data = df[df["phase"] == "optimization"].copy()

    # 根据λ_t值判断阶段
    # λ接近lambda_min: 探索为主
    # λ接近lambda_max: 开发为主
    opt_data["stage"] = pd.cut(
        opt_data["lambda_t"],
        bins=[0, 0.8, 1.5, 3.0],
        labels=["探索主导", "平衡阶段", "开发主导"],
    )

    for stage in ["探索主导", "平衡阶段", "开发主导"]:
        stage_data = opt_data[opt_data["stage"] == stage]
        if len(stage_data) > 0:
            print(f"\n{stage} (n={len(stage_data)}):")
            print(f"  - λ_t 均值: {stage_data['lambda_t'].mean():.3f}")
            print(f"  - r_t 均值: {stage_data['r_t'].mean():.3f}")
            print(f"  - 采集得分: {stage_data['acq_score'].mean():.4f}")


def analyze_parameter_coverage(df: pd.DataFrame):
    """分析参数空间覆盖"""
    print("\n" + "=" * 80)
    print(" 参数空间覆盖分析")
    print("=" * 80)

    for phase in ["initialization", "optimization"]:
        phase_data = df[df["phase"] == phase]
        print(f"\n{phase.capitalize()}:")
        for param in ["x1", "x2", "x3"]:
            values = phase_data[param]
            print(
                f"  - {param}: 范围=[{values.min():.3f}, {values.max():.3f}], "
                f"均值={values.mean():.3f}, 标准差={values.std():.3f}"
            )


def compare_with_random(df: pd.DataFrame, metadata: dict):
    """与随机采样对比"""
    print("\n" + "=" * 80)
    print(" 与随机采样对比")
    print("=" * 80)

    # 计算实际采集策略的性能
    actual_r2 = metadata["evaluation"]["r2"]
    actual_mse = metadata["evaluation"]["mse"]

    print(f"\n采集函数策略:")
    print(f"  - R² = {actual_r2:.4f}")
    print(f"  - MSE = {actual_mse:.4f}")

    # 简单对比(需要运行随机基准才有真实对比)
    print(f"\n说明:")
    print(f"  - 此策略在 {len(df)} 个试次后达到 R²={actual_r2:.4f}")
    print(f"  - 与纯随机采样相比,自适应采集函数通常可以:")
    print(f"    • 用更少样本达到相同精度")
    print(f"    • 更好地探索高不确定性区域")
    print(f"    • 在探索-开发之间实现动态平衡")


def plot_learning_curves(df: pd.DataFrame, timestamp: str, output_dir: str = "results"):
    """绘制学习曲线"""
    print("\n生成额外分析图表...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("学习曲线分析", fontsize=16, fontweight="bold")

    # 1. 累积MSE
    ax = axes[0, 0]
    cumulative_mse = []
    for i in range(len(df)):
        errors = (df["true_value"].iloc[: i + 1] - df["response"].iloc[: i + 1]) ** 2
        cumulative_mse.append(errors.mean())

    ax.plot(cumulative_mse, "b-", linewidth=2)
    ax.axvline(20, color="red", linestyle="--", alpha=0.5, label="优化开始")
    ax.set_xlabel("试次")
    ax.set_ylabel("累积 MSE")
    ax.set_title("模型误差随时间变化")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 响应分布对比
    ax = axes[0, 1]
    init_responses = df[df["phase"] == "initialization"]["response"]
    opt_responses = df[df["phase"] == "optimization"]["response"]

    ax.hist(init_responses, bins=15, alpha=0.6, label="初始化", color="blue")
    ax.hist(opt_responses, bins=15, alpha=0.6, label="优化", color="green")
    ax.set_xlabel("响应值")
    ax.set_ylabel("频数")
    ax.set_title("响应分布对比")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 参数空间分布
    ax = axes[1, 0]
    init_data = df[df["phase"] == "initialization"]
    opt_data = df[df["phase"] == "optimization"]

    ax.scatter(
        init_data["x1"], init_data["x2"], c="blue", alpha=0.5, s=50, label="初始化"
    )
    ax.scatter(opt_data["x1"], opt_data["x2"], c="green", alpha=0.5, s=50, label="优化")
    ax.set_xlabel("x1 (intensity)")
    ax.set_ylabel("x2 (duration)")
    ax.set_title("参数空间采样分布")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 采集得分演化
    ax = axes[1, 1]
    opt_data = df[df["phase"] == "optimization"]
    ax.plot(opt_data.index, opt_data["acq_score"], "g-", linewidth=2)
    ax.set_xlabel("试次")
    ax.set_ylabel("采集得分")
    ax.set_title("采集函数得分演化")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"learning_curves_{timestamp}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ 学习曲线已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print(" 实验结果分析")
    print("=" * 80)

    # 加载数据
    data, timestamp = load_latest_results()
    df = data["trials"]
    metadata = data["metadata"]

    print(f"\n实验概况:")
    print(f"  - 总试次: {len(df)}")
    print(f"  - 初始化: {metadata['n_init']}")
    print(f"  - 优化: {metadata['n_opt']}")
    print(f"  - 被试类型: {metadata['subject_type']}")
    print(f"  - 观测噪声: σ={metadata['subject_noise']}")

    # 执行各项分析
    analyze_adaptive_behavior(df)
    analyze_exploration_exploitation(df)
    analyze_parameter_coverage(df)
    compare_with_random(df, metadata)

    # 生成额外图表
    plot_learning_curves(df, timestamp)

    print("\n" + "=" * 80)
    print(" 分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()

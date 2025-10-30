"""
对比分析 V1 vs V2 实验结果

比较原始采集函数和改进后采集函数的表现
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
from pathlib import Path
import json


def load_experiment_data(results_dir):
    """加载实验数据"""
    results_dir = Path(results_dir)

    # 找到最新的CSV文件
    csv_files = list(results_dir.glob("trial_data*.csv"))
    if not csv_files:
        return None, None

    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    df = pd.DataFrame(pd.read_csv(latest_csv))

    # 找到对应的metadata
    json_files = list(results_dir.glob("metadata*.json"))
    if json_files:
        latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
        with open(latest_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return df, metadata


def compare_coverage(df_v1, df_v2):
    """对比空间覆盖情况"""
    # 计算唯一设计数
    design_cols = ["color_scheme", "layout", "font_size", "animation"]

    unique_v1 = df_v1[design_cols].drop_duplicates()
    unique_v2 = df_v2[design_cols].drop_duplicates()

    n_unique_v1 = len(unique_v1)
    n_unique_v2 = len(unique_v2)

    coverage_v1 = n_unique_v1 / 360.0
    coverage_v2 = n_unique_v2 / 360.0

    print(f"\n{'='*60}")
    print("空间覆盖对比")
    print(f"{'='*60}")
    print(f"V1 (原始):")
    print(f"  唯一设计数: {n_unique_v1}/360 ({coverage_v1*100:.1f}%)")
    print(f"  平均重复次数: {len(df_v1)/n_unique_v1:.2f}")
    print(f"\nV2 (改进):")
    print(f"  唯一设计数: {n_unique_v2}/360 ({coverage_v2*100:.1f}%)")
    print(f"  平均重复次数: {len(df_v2)/n_unique_v2:.2f}")
    print(f"\n改进幅度: {(n_unique_v2-n_unique_v1)/n_unique_v1*100:+.1f}%")

    return {
        "n_unique_v1": n_unique_v1,
        "n_unique_v2": n_unique_v2,
        "coverage_v1": coverage_v1,
        "coverage_v2": coverage_v2,
    }


def compare_distribution(df_v1, df_v2):
    """对比分数分布"""
    print(f"\n{'='*60}")
    print("分数分布对比")
    print(f"{'='*60}")
    print(f"V1 (原始):")
    print(f"  均值: {df_v1['true_score'].mean():.2f}")
    print(f"  标准差: {df_v1['true_score'].std():.2f}")
    print(f"  最小值: {df_v1['true_score'].min():.2f}")
    print(f"  最大值: {df_v1['true_score'].max():.2f}")
    print(f"\nV2 (改进):")
    print(f"  均值: {df_v2['true_score'].mean():.2f}")
    print(f"  标准差: {df_v2['true_score'].std():.2f}")
    print(f"  最小值: {df_v2['true_score'].min():.2f}")
    print(f"  最大值: {df_v2['true_score'].max():.2f}")

    return {
        "mean_v1": df_v1["true_score"].mean(),
        "mean_v2": df_v2["true_score"].mean(),
        "std_v1": df_v1["true_score"].std(),
        "std_v2": df_v2["true_score"].std(),
    }


def compare_high_score_discovery(df_v1, df_v2):
    """对比高分设计发现率"""
    # 全空间最高分是10.50
    thresholds = [9.5, 9.8, 10.0]

    print(f"\n{'='*60}")
    print("高分设计发现率对比")
    print(f"{'='*60}")

    for threshold in thresholds:
        count_v1 = (df_v1["true_score"] >= threshold).sum()
        count_v2 = (df_v2["true_score"] >= threshold).sum()

        print(f"\nTrue Score >= {threshold}:")
        print(f"  V1: {count_v1} 个")
        print(f"  V2: {count_v2} 个")
        if count_v1 > 0:
            print(f"  改进: {(count_v2-count_v1)/count_v1*100:+.1f}%")

    return {
        "high_score_v1": (df_v1["true_score"] >= 9.5).sum(),
        "high_score_v2": (df_v2["true_score"] >= 9.5).sum(),
    }


def create_comparison_plots(df_v1, df_v2, coverage_stats, output_path):
    """创建对比图表"""
    fig = plt.figure(figsize=(20, 12))

    # 1. 唯一设计数对比 (柱状图)
    ax1 = plt.subplot(2, 4, 1)
    versions = ["V1\n(原始)", "V2\n(改进)"]
    unique_counts = [coverage_stats["n_unique_v1"], coverage_stats["n_unique_v2"]]
    colors = ["#FF6B6B", "#4ECDC4"]
    bars = ax1.bar(
        versions, unique_counts, color=colors, edgecolor="black", linewidth=2
    )
    ax1.set_ylabel("唯一设计数", fontsize=12, fontweight="bold")
    ax1.set_title("1. 唯一设计数对比", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    # 添加数值标签
    for bar, count in zip(bars, unique_counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({count/360*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax1.set_ylim(0, max(unique_counts) * 1.2)

    # 2. 分数分布对比 (直方图)
    ax2 = plt.subplot(2, 4, 2)
    ax2.hist(
        df_v1["true_score"],
        bins=20,
        alpha=0.5,
        label="V1",
        color="#FF6B6B",
        edgecolor="black",
    )
    ax2.hist(
        df_v2["true_score"],
        bins=20,
        alpha=0.5,
        label="V2",
        color="#4ECDC4",
        edgecolor="black",
    )
    ax2.set_xlabel("True Score", fontsize=12)
    ax2.set_ylabel("频数", fontsize=12)
    ax2.set_title("2. 分数分布对比", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    # 3. 累积分布函数 (CDF)
    ax3 = plt.subplot(2, 4, 3)
    sorted_v1 = np.sort(df_v1["true_score"])
    sorted_v2 = np.sort(df_v2["true_score"])
    cdf_v1 = np.arange(1, len(sorted_v1) + 1) / len(sorted_v1)
    cdf_v2 = np.arange(1, len(sorted_v2) + 1) / len(sorted_v2)
    ax3.plot(sorted_v1, cdf_v1, label="V1", linewidth=2, color="#FF6B6B")
    ax3.plot(sorted_v2, cdf_v2, label="V2", linewidth=2, color="#4ECDC4")
    ax3.set_xlabel("True Score", fontsize=12)
    ax3.set_ylabel("累积概率", fontsize=12)
    ax3.set_title("3. 累积分布函数 (CDF)", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)

    # 4. 箱线图对比
    ax4 = plt.subplot(2, 4, 4)
    data_to_plot = [df_v1["true_score"], df_v2["true_score"]]
    bp = ax4.boxplot(data_to_plot, labels=["V1", "V2"], patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel("True Score", fontsize=12)
    ax4.set_title("4. 分数箱线图对比", fontsize=14, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    # 5. 变量覆盖对比 - Color Scheme
    ax5 = plt.subplot(2, 4, 5)
    color_v1 = df_v1["color_scheme"].value_counts().sort_index()
    color_v2 = df_v2["color_scheme"].value_counts().sort_index()
    x = np.arange(len(color_v1))
    width = 0.35
    ax5.bar(
        x - width / 2,
        color_v1.values,
        width,
        label="V1",
        color="#FF6B6B",
        alpha=0.8,
        edgecolor="black",
    )
    ax5.bar(
        x + width / 2,
        color_v2.values,
        width,
        label="V2",
        color="#4ECDC4",
        alpha=0.8,
        edgecolor="black",
    )
    ax5.set_xlabel("Color Scheme", fontsize=12)
    ax5.set_ylabel("采样次数", fontsize=12)
    ax5.set_title("5. Color Scheme 覆盖对比", fontsize=14, fontweight="bold")
    ax5.set_xticks(x)
    ax5.set_xticklabels(color_v1.index, rotation=45, ha="right")
    ax5.legend(fontsize=10)
    ax5.grid(axis="y", alpha=0.3)

    # 6. 变量覆盖对比 - Layout
    ax6 = plt.subplot(2, 4, 6)
    layout_v1 = df_v1["layout"].value_counts().sort_index()
    layout_v2 = df_v2["layout"].value_counts().sort_index()
    x = np.arange(len(layout_v1))
    ax6.bar(
        x - width / 2,
        layout_v1.values,
        width,
        label="V1",
        color="#FF6B6B",
        alpha=0.8,
        edgecolor="black",
    )
    ax6.bar(
        x + width / 2,
        layout_v2.values,
        width,
        label="V2",
        color="#4ECDC4",
        alpha=0.8,
        edgecolor="black",
    )
    ax6.set_xlabel("Layout", fontsize=12)
    ax6.set_ylabel("采样次数", fontsize=12)
    ax6.set_title("6. Layout 覆盖对比", fontsize=14, fontweight="bold")
    ax6.set_xticks(x)
    ax6.set_xticklabels(layout_v1.index, rotation=45, ha="right")
    ax6.legend(fontsize=10)
    ax6.grid(axis="y", alpha=0.3)

    # 7. True Score 时间序列对比
    ax7 = plt.subplot(2, 4, 7)
    ax7.plot(
        df_v1["trial"],
        df_v1["true_score"],
        "o-",
        label="V1",
        alpha=0.6,
        color="#FF6B6B",
        markersize=4,
    )
    ax7.plot(
        df_v2["trial"],
        df_v2["true_score"],
        "s-",
        label="V2",
        alpha=0.6,
        color="#4ECDC4",
        markersize=4,
    )
    ax7.set_xlabel("Trial", fontsize=12)
    ax7.set_ylabel("True Score", fontsize=12)
    ax7.set_title("7. True Score 时间序列", fontsize=14, fontweight="bold")
    ax7.legend(fontsize=10)
    ax7.grid(alpha=0.3)
    ax7.axvline(
        x=20, color="red", linestyle="--", linewidth=2, alpha=0.5, label="主动学习开始"
    )

    # 8. 高分区域发现对比
    ax8 = plt.subplot(2, 4, 8)
    thresholds = [8.0, 8.5, 9.0, 9.5, 10.0]
    count_v1 = [(df_v1["true_score"] >= t).sum() for t in thresholds]
    count_v2 = [(df_v2["true_score"] >= t).sum() for t in thresholds]
    x = np.arange(len(thresholds))
    ax8.plot(x, count_v1, "o-", label="V1", linewidth=2, markersize=8, color="#FF6B6B")
    ax8.plot(x, count_v2, "s-", label="V2", linewidth=2, markersize=8, color="#4ECDC4")
    ax8.set_xlabel("Score 阈值", fontsize=12)
    ax8.set_ylabel("发现数量", fontsize=12)
    ax8.set_title("8. 高分设计发现率", fontsize=14, fontweight="bold")
    ax8.set_xticks(x)
    ax8.set_xticklabels([f"≥{t}" for t in thresholds])
    ax8.legend(fontsize=10)
    ax8.grid(alpha=0.3)

    plt.suptitle("V1 vs V2 采集函数对比分析", fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 对比图表已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print(" V1 vs V2 采集函数对比分析")
    print("=" * 80)

    # 加载数据
    print("\n加载实验数据...")
    df_v1, meta_v1 = load_experiment_data("results")
    df_v2, meta_v2 = load_experiment_data("results_v2")

    if df_v1 is None or df_v2 is None:
        print("错误: 无法找到实验数据文件")
        return

    print(f"✓ V1数据: {len(df_v1)} trials")
    print(f"✓ V2数据: {len(df_v2)} trials")

    # 对比分析
    coverage_stats = compare_coverage(df_v1, df_v2)
    distribution_stats = compare_distribution(df_v1, df_v2)
    high_score_stats = compare_high_score_discovery(df_v1, df_v2)

    # 创建对比图表
    print(f"\n{'='*60}")
    print("生成对比图表...")
    print(f"{'='*60}")
    output_path = Path("report") / "comparison_v1_vs_v2.png"
    output_path.parent.mkdir(exist_ok=True)
    create_comparison_plots(df_v1, df_v2, coverage_stats, output_path)

    # 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    improvement_coverage = (
        (coverage_stats["n_unique_v2"] - coverage_stats["n_unique_v1"])
        / coverage_stats["n_unique_v1"]
        * 100
    )
    print(f"\n1. 空间覆盖: {improvement_coverage:+.1f}% 改进")
    print(f"   V1: {coverage_stats['n_unique_v1']} 个唯一设计")
    print(f"   V2: {coverage_stats['n_unique_v2']} 个唯一设计")

    improvement_high_score = (
        (high_score_stats["high_score_v2"] - high_score_stats["high_score_v1"])
        / max(high_score_stats["high_score_v1"], 1)
        * 100
    )
    print(f"\n2. 高分发现: {improvement_high_score:+.1f}% 改进")
    print(f"   V1: {high_score_stats['high_score_v1']} 个 (≥9.5)")
    print(f"   V2: {high_score_stats['high_score_v2']} 个 (≥9.5)")

    print(f"\n3. 平均分数:")
    print(
        f"   V1: {distribution_stats['mean_v1']:.2f} ± {distribution_stats['std_v1']:.2f}"
    )
    print(
        f"   V2: {distribution_stats['mean_v2']:.2f} ± {distribution_stats['std_v2']:.2f}"
    )

    print(f"\n{'='*60}")
    print("✓ 对比分析完成!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

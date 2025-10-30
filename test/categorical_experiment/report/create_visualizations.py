"""
可视化报告生成脚本
创建详细的统计图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据
data_dir = Path(__file__).parent
full_space_df = pd.read_csv(data_dir / "full_design_space_analysis.csv")
with open(data_dir / "statistical_results.json", "r") as f:
    stats_results = json.load(f)

trial_data_dir = Path(__file__).parent.parent / "results"
trial_data = pd.read_csv(trial_data_dir / "trial_data_20251030_000437.csv")

# 创建大图
fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)

# ==================== 图1: 分数分布对比 ====================
ax1 = fig.add_subplot(gs[0, :])

# 准备数据
sampled_scores = full_space_df[full_space_df["sampled"]]["true_score"]
unsampled_scores = full_space_df[~full_space_df["sampled"]]["true_score"]

# 绘制直方图
bins = np.linspace(6, 11, 30)
ax1.hist(
    full_space_df["true_score"],
    bins=bins,
    alpha=0.3,
    label="全空间 (360)",
    density=True,
    color="gray",
    edgecolor="black",
)
ax1.hist(
    sampled_scores,
    bins=bins,
    alpha=0.6,
    label="已采样 (23)",
    density=True,
    color="blue",
    edgecolor="darkblue",
)

# 添加均值线
ax1.axvline(
    full_space_df["true_score"].mean(),
    color="gray",
    linestyle="--",
    linewidth=2,
    label=f'全空间均值: {full_space_df["true_score"].mean():.2f}',
)
ax1.axvline(
    sampled_scores.mean(),
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f"采样均值: {sampled_scores.mean():.2f}",
)

ax1.set_xlabel("真实分数", fontsize=12)
ax1.set_ylabel("密度", fontsize=12)
ax1.set_title("图1: 分数分布对比 - 采样 vs 全空间", fontsize=14, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 添加KS检验结果
ks_text = f'KS检验: p={stats_results["distribution"]["ks_pvalue"]:.3f}\n'
ks_text += (
    "分布相似 (p>0.05)"
    if stats_results["distribution"]["ks_pvalue"] > 0.05
    else "分布差异 (p<0.05)"
)
ax1.text(
    0.02,
    0.98,
    ks_text,
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

# ==================== 图2: 分位数对比 ====================
ax2 = fig.add_subplot(gs[1, 0])

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
q_full = [full_space_df["true_score"].quantile(q) for q in quantiles]
q_sampled = [sampled_scores.quantile(q) for q in quantiles]
q_labels = ["10%", "25%", "50%", "75%", "90%"]

x = np.arange(len(quantiles))
width = 0.35

ax2.bar(x - width / 2, q_full, width, label="全空间", color="gray", alpha=0.7)
ax2.bar(x + width / 2, q_sampled, width, label="已采样", color="blue", alpha=0.7)

ax2.set_xlabel("分位数", fontsize=11)
ax2.set_ylabel("分数", fontsize=11)
ax2.set_title("图2: 分位数对比", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(q_labels)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

# ==================== 图3: 预测误差分布 ====================
ax3 = fig.add_subplot(gs[1, 1])

errors = trial_data["rating"] - trial_data["true_score"]

ax3.hist(errors, bins=30, edgecolor="black", alpha=0.7, color="coral")
ax3.axvline(
    errors.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"平均误差: {errors.mean():.2f}",
)
ax3.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)

ax3.set_xlabel("预测误差 (观测 - 真实)", fontsize=11)
ax3.set_ylabel("频数", fontsize=11)
ax3.set_title("图3: 预测误差分布", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")

# 添加统计信息
mae = stats_results["errors"]["mae"]
rmse = stats_results["errors"]["rmse"]
error_text = f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}"
ax3.text(
    0.02,
    0.98,
    error_text,
    transform=ax3.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
)

# ==================== 图4: 分数区间覆盖 ====================
ax4 = fig.add_subplot(gs[1, 2])

score_bins = pd.cut(full_space_df["true_score"], bins=[6, 7, 8, 9, 10, 11])
bin_coverage = []
bin_labels = []

for bin_range in score_bins.cat.categories:
    n_full = (full_space_df["score_bin"] == bin_range).sum()
    n_sampled = (
        full_space_df[full_space_df["sampled"]]["score_bin"] == bin_range
    ).sum()
    rate = n_sampled / n_full * 100 if n_full > 0 else 0
    bin_coverage.append(rate)
    bin_labels.append(str(bin_range))

# 重新分组score_bin
full_space_df["score_bin"] = score_bins

ax4.barh(bin_labels, bin_coverage, color="teal", alpha=0.7)
ax4.set_xlabel("采样率 (%)", fontsize=11)
ax4.set_ylabel("分数区间", fontsize=11)
ax4.set_title("图4: 各分数区间采样覆盖率", fontsize=12, fontweight="bold")
ax4.grid(True, alpha=0.3, axis="x")

# ==================== 图5-8: 变量主效应 ====================
variables = ["color_scheme", "layout", "font_size", "animation"]
titles = [
    "图5: 色彩方案主效应",
    "图6: 布局主效应",
    "图7: 字体大小主效应",
    "图8: 动画效应",
]
positions = [(2, 0), (2, 1), (2, 2), (3, 0)]

for var, title, pos in zip(variables, titles, positions):
    ax = fig.add_subplot(gs[pos[0], pos[1]])

    # 全空间均值
    full_means = (
        full_space_df.groupby(var)["true_score"].mean().sort_values(ascending=False)
    )
    # 采样均值
    sampled_df = full_space_df[full_space_df["sampled"]]
    if len(sampled_df) > 0:
        sampled_means = sampled_df.groupby(var)["true_score"].mean()
        # 对齐顺序
        sampled_means = sampled_means.reindex(full_means.index, fill_value=np.nan)
    else:
        sampled_means = pd.Series(np.nan, index=full_means.index)

    x = np.arange(len(full_means))
    width = 0.35

    ax.bar(
        x - width / 2, full_means.values, width, label="全空间", color="gray", alpha=0.7
    )
    ax.bar(
        x + width / 2,
        sampled_means.values,
        width,
        label="已采样",
        color="green",
        alpha=0.7,
    )

    ax.set_ylabel("平均真实分数", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(full_means.index, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

# ==================== 图9: 高分设计发现 ====================
ax9 = fig.add_subplot(gs[3, 1])

percentiles = [90, 95, 99]
discovery_rates = []
labels = ["Top 10%", "Top 5%", "Top 1%"]

for pct in percentiles:
    threshold = full_space_df["true_score"].quantile(pct / 100)
    n_full = (full_space_df["true_score"] >= threshold).sum()
    n_sampled = (
        full_space_df[full_space_df["sampled"]]["true_score"] >= threshold
    ).sum()
    rate = n_sampled / n_full * 100 if n_full > 0 else 0
    discovery_rates.append(rate)

ax9.bar(
    labels,
    discovery_rates,
    color="gold",
    alpha=0.7,
    edgecolor="darkorange",
    linewidth=2,
)
ax9.set_ylabel("发现率 (%)", fontsize=11)
ax9.set_title("图9: 高分设计发现率", fontsize=12, fontweight="bold")
ax9.grid(True, alpha=0.3, axis="y")
ax9.set_ylim(0, max(discovery_rates) * 1.2)

# 添加数值标签
for i, v in enumerate(discovery_rates):
    ax9.text(i, v + max(discovery_rates) * 0.02, f"{v:.1f}%", ha="center", fontsize=10)

# ==================== 图10: 观测 vs 真实散点图 ====================
ax10 = fig.add_subplot(gs[3, 2])

ax10.scatter(
    trial_data["true_score"],
    trial_data["rating"],
    alpha=0.5,
    s=50,
    c=trial_data["trial"],
    cmap="viridis",
)
ax10.plot([6, 11], [6, 11], "r--", linewidth=2, label="完美预测")

# 拟合线
z = np.polyfit(trial_data["true_score"], trial_data["rating"], 1)
p = np.poly1d(z)
x_line = np.linspace(
    trial_data["true_score"].min(), trial_data["true_score"].max(), 100
)
ax10.plot(
    x_line,
    p(x_line),
    "b-",
    linewidth=2,
    alpha=0.7,
    label=f"拟合线: y={z[0]:.2f}x+{z[1]:.2f}",
)

ax10.set_xlabel("真实分数", fontsize=11)
ax10.set_ylabel("观测评分", fontsize=11)
ax10.set_title("图10: 观测 vs 真实分数", fontsize=12, fontweight="bold")
ax10.legend(fontsize=9)
ax10.grid(True, alpha=0.3)

# 添加相关系数
corr = np.corrcoef(trial_data["true_score"], trial_data["rating"])[0, 1]
ax10.text(
    0.02,
    0.98,
    f"r = {corr:.3f}",
    transform=ax10.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
)

# ==================== 图11: 采样空间可视化 ====================
ax11 = fig.add_subplot(gs[4, :2])

# 创建2D投影：color_scheme vs layout
pivot_data = (
    full_space_df.groupby(["color_scheme", "layout"])
    .agg({"true_score": "mean", "sampled": "any"})
    .reset_index()
)

pivot_matrix = pivot_data.pivot(
    index="color_scheme", columns="layout", values="true_score"
)
pivot_sampled = pivot_data.pivot(
    index="color_scheme", columns="layout", values="sampled"
)

im = ax11.imshow(pivot_matrix, cmap="RdYlGn", aspect="auto", vmin=6, vmax=11)

# 标记采样点
for i in range(len(pivot_matrix.index)):
    for j in range(len(pivot_matrix.columns)):
        if pivot_sampled.iloc[i, j]:
            ax11.plot(
                j, i, "b*", markersize=20, markeredgecolor="black", markeredgewidth=1
            )

ax11.set_xticks(np.arange(len(pivot_matrix.columns)))
ax11.set_yticks(np.arange(len(pivot_matrix.index)))
ax11.set_xticklabels(pivot_matrix.columns, rotation=45, ha="right")
ax11.set_yticklabels(pivot_matrix.index)
ax11.set_xlabel("布局", fontsize=11)
ax11.set_ylabel("色彩方案", fontsize=11)
ax11.set_title("图11: 设计空间采样可视化 (蓝星=已采样)", fontsize=12, fontweight="bold")

# 添加colorbar
cbar = plt.colorbar(im, ax=ax11)
cbar.set_label("平均真实分数", fontsize=10)

# ==================== 图12: 综合评分雷达图 ====================
ax12 = fig.add_subplot(gs[4, 2], projection="polar")

categories = ["覆盖率", "分布\n一致性", "预测\n准确性", "相关性", "高分\n发现"]
values = [
    stats_results["scores"]["coverage_score"],
    stats_results["scores"]["distribution_score"],
    stats_results["scores"]["error_score"],
    stats_results["scores"]["correlation_score"],
    stats_results["scores"]["discovery_score"],
]

# 闭合雷达图
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

ax12.plot(angles, values, "o-", linewidth=2, color="blue", label="实际得分")
ax12.fill(angles, values, alpha=0.25, color="blue")

# 添加参考线
ax12.plot(
    angles,
    [0.7] * len(angles),
    "--",
    linewidth=1,
    color="green",
    alpha=0.5,
    label="良好阈值",
)
ax12.plot(
    angles,
    [0.5] * len(angles),
    "--",
    linewidth=1,
    color="orange",
    alpha=0.5,
    label="及格阈值",
)

ax12.set_xticks(angles[:-1])
ax12.set_xticklabels(categories, fontsize=9)
ax12.set_ylim(0, 1)
ax12.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax12.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
ax12.set_title("图12: 综合评分雷达图", fontsize=12, fontweight="bold", pad=20)
ax12.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)
ax12.grid(True)

# ==================== 图13: 采样效率分析 ====================
ax13 = fig.add_subplot(gs[5, :])

# 按trial顺序查看分数趋势
trial_data_sorted = trial_data.sort_values("trial")
init_data = trial_data_sorted[trial_data_sorted["phase"] == "initialization"]
opt_data = trial_data_sorted[trial_data_sorted["phase"] == "optimization"]

# 计算移动平均
window = 5
if len(init_data) >= window:
    init_ma = init_data["true_score"].rolling(window=window, min_periods=1).mean()
else:
    init_ma = init_data["true_score"]

if len(opt_data) >= window:
    opt_ma = opt_data["true_score"].rolling(window=window, min_periods=1).mean()
else:
    opt_ma = opt_data["true_score"]

# 绘制真实分数趋势
ax13.scatter(
    init_data["trial"],
    init_data["true_score"],
    alpha=0.4,
    s=30,
    c="orange",
    label="初始化 (真实分数)",
)
ax13.scatter(
    opt_data["trial"],
    opt_data["true_score"],
    alpha=0.4,
    s=30,
    c="blue",
    label="优化 (真实分数)",
)

# 移动平均线
ax13.plot(
    init_data["trial"],
    init_ma,
    color="orange",
    linewidth=2,
    alpha=0.7,
    label="初始化 (移动平均)",
)
ax13.plot(
    opt_data["trial"],
    opt_ma,
    color="blue",
    linewidth=2,
    alpha=0.7,
    label="优化 (移动平均)",
)

# 添加全空间均值参考线
ax13.axhline(
    full_space_df["true_score"].mean(),
    color="gray",
    linestyle="--",
    linewidth=2,
    label=f'全空间均值: {full_space_df["true_score"].mean():.2f}',
)

# 分隔线
ax13.axvline(20, color="red", linestyle=":", linewidth=2, alpha=0.5)
ax13.text(
    20, ax13.get_ylim()[1] * 0.95, "优化阶段开始", ha="center", fontsize=10, color="red"
)

ax13.set_xlabel("Trial", fontsize=11)
ax13.set_ylabel("真实分数", fontsize=11)
ax13.set_title(
    "图13: 采样效率分析 - 真实分数随时间变化", fontsize=12, fontweight="bold"
)
ax13.legend(fontsize=9, loc="lower right")
ax13.grid(True, alpha=0.3)

# 整体标题
fig.suptitle(
    "深度统计分析报告：采样质量评估\n80个采样点 vs 360个设计组合",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

# 保存
output_path = data_dir / "visualization_report.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n✓ 可视化报告已保存: {output_path}")
plt.close()

print("✓ 所有图表生成完成！")

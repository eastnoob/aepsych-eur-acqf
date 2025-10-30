"""
深度统计分析：采样质量评估
分析采样数据相较于真实模型的误差和代表性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kstest, anderson, shapiro, mannwhitneyu
import json
from pathlib import Path
import sys

# 添加路径以导入虚拟用户
sys.path.insert(0, str(Path(__file__).parent.parent))
from virtual_user import VirtualUser

# 设置中文字体和样式
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

# 加载数据
data_dir = Path(__file__).parent.parent / "results"
trial_data = pd.read_csv(data_dir / "trial_data_20251030_000437.csv")
with open(data_dir / "metadata_20251030_000437.json", "r") as f:
    metadata = json.load(f)

print("=" * 80)
print("深度统计分析：采样质量评估")
print("=" * 80)
print(f"数据集: 80个采样点 / 360个总设计组合 (22.2%)")
print()

# ==================== 1. 生成完整设计空间的真实分数 ====================
print("1. 生成完整设计空间 (360个组合)...")

# 创建虚拟用户（与实验相同的配置）
user = VirtualUser(user_type="balanced", noise_level=0.0, seed=42)  # 无噪声的真实分数

# 生成所有可能的组合
color_schemes = ["blue", "green", "red", "purple", "orange"]
layouts = ["grid", "list", "card", "timeline"]
font_sizes = [12, 14, 16, 18, 20, 22]
animations = ["none", "subtle", "dynamic"]

all_designs = []
for color in color_schemes:
    for layout in layouts:
        for font in font_sizes:
            for anim in animations:
                design = {
                    "color_scheme": color,
                    "layout": layout,
                    "font_size": font,
                    "animation": anim,
                }
                # 计算真实分数（无噪声）
                true_score = user.get_ground_truth(design)
                all_designs.append({**design, "true_score": true_score})

full_space_df = pd.DataFrame(all_designs)
print(f"✓ 生成 {len(full_space_df)} 个设计的真实分数")
print()

# ==================== 2. 采样覆盖分析 ====================
print("=" * 80)
print("2. 采样覆盖分析")
print("=" * 80)

# 标记哪些设计被采样
sampled_designs = trial_data[
    ["color_scheme", "layout", "font_size", "animation"]
].copy()
full_space_df["sampled"] = False

for idx, row in sampled_designs.iterrows():
    mask = (
        (full_space_df["color_scheme"] == row["color_scheme"])
        & (full_space_df["layout"] == row["layout"])
        & (full_space_df["font_size"] == row["font_size"])
        & (full_space_df["animation"] == row["animation"])
    )
    full_space_df.loc[mask, "sampled"] = True

n_sampled = full_space_df["sampled"].sum()
print(f"唯一设计采样: {n_sampled}/360 ({n_sampled/360*100:.1f}%)")
print()

# 分数分布对比
sampled_scores = full_space_df[full_space_df["sampled"]]["true_score"]
unsampled_scores = full_space_df[~full_space_df["sampled"]]["true_score"]

print("分数分布统计:")
print(
    f"  全空间: μ={full_space_df['true_score'].mean():.3f}, σ={full_space_df['true_score'].std():.3f}"
)
print(f"  已采样: μ={sampled_scores.mean():.3f}, σ={sampled_scores.std():.3f}")
print(f"  未采样: μ={unsampled_scores.mean():.3f}, σ={unsampled_scores.std():.3f}")
print()

# ==================== 3. 分布一致性检验 ====================
print("=" * 80)
print("3. 分布一致性检验")
print("=" * 80)

# 3.1 Kolmogorov-Smirnov检验
ks_stat, ks_pvalue = kstest(
    sampled_scores,
    lambda x: stats.percentileofscore(full_space_df["true_score"], x) / 100,
)
print("3.1 Kolmogorov-Smirnov检验 (采样 vs 全空间)")
print(f"  KS统计量: {ks_stat:.4f}")
print(f"  p-value: {ks_pvalue:.4f}")
print(f"  结论: {'分布相似 (p>0.05)' if ks_pvalue > 0.05 else '分布差异显著 (p<0.05)'}")
print()

# 3.2 Mann-Whitney U检验 (非参数检验)
u_stat, u_pvalue = mannwhitneyu(
    sampled_scores, unsampled_scores, alternative="two-sided"
)
print("3.2 Mann-Whitney U检验 (采样 vs 未采样)")
print(f"  U统计量: {u_stat:.2f}")
print(f"  p-value: {u_pvalue:.4f}")
print(
    f"  结论: {'中位数无显著差异 (p>0.05)' if u_pvalue > 0.05 else '中位数有显著差异 (p<0.05)'}"
)
print()

# 3.3 分位数对比
print("3.3 分位数对比")
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
print("  分位数    全空间    已采样    差异")
print("  " + "-" * 40)
for q in quantiles:
    q_full = full_space_df["true_score"].quantile(q)
    q_sampled = sampled_scores.quantile(q)
    diff = q_sampled - q_full
    print(f"  {q:4.0%}      {q_full:6.3f}    {q_sampled:6.3f}    {diff:+.3f}")
print()

# ==================== 4. 预测误差分析 ====================
print("=" * 80)
print("4. 预测误差分析 (观测评分 vs 真实分数)")
print("=" * 80)

# 合并数据
trial_data_with_true = trial_data.merge(
    full_space_df[["color_scheme", "layout", "font_size", "animation", "true_score"]],
    on=["color_scheme", "layout", "font_size", "animation"],
    how="left",
    suffixes=("_observed", "_true"),
)

# 注意：trial_data已经有true_score列了
errors = trial_data["rating"] - trial_data["true_score"]

print(f"误差统计 (观测评分 - 真实分数):")
print(f"  平均误差 (ME):  {errors.mean():.3f}")
print(f"  平均绝对误差 (MAE): {np.abs(errors).mean():.3f}")
print(f"  均方根误差 (RMSE): {np.sqrt((errors**2).mean()):.3f}")
print(f"  标准差: {errors.std():.3f}")
print(f"  中位数误差: {errors.median():.3f}")
print()

# 误差分布
print("误差分布:")
for threshold in [0.5, 1.0, 1.5, 2.0]:
    pct = (np.abs(errors) <= threshold).mean() * 100
    print(f"  ±{threshold}以内: {pct:.1f}%")
print()

# ==================== 5. 采样偏差分析 ====================
print("=" * 80)
print("5. 采样偏差分析")
print("=" * 80)

# 5.1 分数区间覆盖
score_bins = [6, 7, 8, 9, 10, 11]
full_space_df["score_bin"] = pd.cut(full_space_df["true_score"], bins=score_bins)

print("5.1 分数区间覆盖")
print("  区间        全空间   已采样   采样率")
print("  " + "-" * 45)
for bin_range in full_space_df["score_bin"].cat.categories:
    n_full = (full_space_df["score_bin"] == bin_range).sum()
    n_sampled = (
        full_space_df[full_space_df["sampled"]]["score_bin"] == bin_range
    ).sum()
    rate = n_sampled / n_full * 100 if n_full > 0 else 0
    print(f"  {bin_range}    {n_full:4d}    {n_sampled:4d}    {rate:5.1f}%")
print()

# 5.2 高分设计发现率
top_percentiles = [90, 95, 99]
print("5.2 高分设计发现率")
print("  百分位    阈值    全空间   已采样   发现率")
print("  " + "-" * 50)
for pct in top_percentiles:
    threshold = full_space_df["true_score"].quantile(pct / 100)
    n_full = (full_space_df["true_score"] >= threshold).sum()
    n_sampled = (
        full_space_df[full_space_df["sampled"]]["true_score"] >= threshold
    ).sum()
    rate = n_sampled / n_full * 100 if n_full > 0 else 0
    print(
        f"  Top {100-pct}%    {threshold:5.2f}    {n_full:4d}    {n_sampled:4d}    {rate:5.1f}%"
    )
print()

# ==================== 6. 变量效应分析 ====================
print("=" * 80)
print("6. 变量主效应分析")
print("=" * 80)

variables = ["color_scheme", "layout", "font_size", "animation"]

for var in variables:
    print(f"\n{var}:")
    # 全空间
    full_means = full_space_df.groupby(var)["true_score"].agg(["mean", "count"])
    # 采样空间
    sampled_means = (
        full_space_df[full_space_df["sampled"]]
        .groupby(var)["true_score"]
        .agg(["mean", "count"])
    )

    print(f"  水平       全空间均值  采样均值  样本数  差异")
    print("  " + "-" * 55)
    for level in full_means.index:
        full_mean = full_means.loc[level, "mean"]
        if level in sampled_means.index:
            samp_mean = sampled_means.loc[level, "mean"]
            samp_count = sampled_means.loc[level, "count"]
            diff = samp_mean - full_mean
            print(
                f"  {str(level):12s} {full_mean:8.3f}    {samp_mean:7.3f}    {samp_count:3.0f}    {diff:+.3f}"
            )
        else:
            print(f"  {str(level):12s} {full_mean:8.3f}    --        0     --")

print()

# ==================== 7. 信息量分析 ====================
print("=" * 80)
print("7. 信息量分析")
print("=" * 80)

# 7.1 采样的信息熵
from scipy.stats import entropy


def compute_entropy(scores):
    """计算分数分布的熵"""
    hist, _ = np.histogram(scores, bins=20, density=True)
    hist = hist[hist > 0]  # 移除0值
    return entropy(hist)


entropy_full = compute_entropy(full_space_df["true_score"])
entropy_sampled = compute_entropy(sampled_scores)

print(f"Shannon熵:")
print(f"  全空间: {entropy_full:.4f}")
print(f"  已采样: {entropy_sampled:.4f}")
print(f"  熵比率: {entropy_sampled/entropy_full:.2%}")
print()

# 7.2 代表性指标
# 使用变异系数 (CV = σ/μ)
cv_full = full_space_df["true_score"].std() / full_space_df["true_score"].mean()
cv_sampled = sampled_scores.std() / sampled_scores.mean()

print(f"变异系数 (CV = σ/μ):")
print(f"  全空间: {cv_full:.4f}")
print(f"  已采样: {cv_sampled:.4f}")
print(f"  差异: {abs(cv_sampled - cv_full)/cv_full*100:.1f}%")
print()

# ==================== 8. 综合评估 ====================
print("=" * 80)
print("8. 综合评估")
print("=" * 80)

# 计算综合指标
coverage_score = n_sampled / 360  # 覆盖率
distribution_score = 1 - abs(ks_stat)  # KS统计量越小越好
error_score = 1 - (metadata["metrics"]["mae"] / 10)  # MAE归一化
correlation_score = metadata["metrics"]["correlation"]  # 相关系数

# 高分发现
top_10pct_threshold = full_space_df["true_score"].quantile(0.9)
top_10pct_found = (
    full_space_df[full_space_df["sampled"]]["true_score"] >= top_10pct_threshold
).sum()
discovery_score = top_10pct_found / (
    len(full_space_df) * 0.1
)  # 发现了多少比例的top 10%

print("综合指标得分 (0-1):")
print(f"  覆盖率:        {coverage_score:.3f}  {'⭐'*int(coverage_score*5)}")
print(f"  分布一致性:    {distribution_score:.3f}  {'⭐'*int(distribution_score*5)}")
print(f"  预测准确性:    {error_score:.3f}  {'⭐'*int(error_score*5)}")
print(f"  相关性:        {correlation_score:.3f}  {'⭐'*int(correlation_score*5)}")
print(f"  高分发现:      {discovery_score:.3f}  {'⭐'*int(discovery_score*5)}")
print()

overall_score = (
    coverage_score
    + distribution_score
    + error_score
    + correlation_score
    + discovery_score
) / 5
print(f"总体评分: {overall_score:.3f} / 1.000  {'⭐'*int(overall_score*5)}")
print()

# ==================== 9. 结论 ====================
print("=" * 80)
print("9. 结论与建议")
print("=" * 80)

conclusions = []

# 覆盖性
if n_sampled / 360 > 0.15:
    conclusions.append("✓ 覆盖性: 良好 - 采样了10.8%的设计空间")
else:
    conclusions.append("✗ 覆盖性: 不足 - 需要增加采样数量")

# 分布一致性
if ks_pvalue > 0.05:
    conclusions.append(
        f"✓ 分布一致性: 优秀 - 采样分布与全空间无显著差异 (p={ks_pvalue:.3f})"
    )
else:
    conclusions.append(
        f"⚠ 分布一致性: 有偏差 - 采样分布与全空间有差异 (p={ks_pvalue:.3f})"
    )

# 预测准确性
if metadata["metrics"]["within_1"] >= 0.8:
    conclusions.append(
        f"✓ 预测准确性: 优秀 - {metadata['metrics']['within_1']*100:.0f}%的预测在±1以内"
    )
else:
    conclusions.append(
        f"⚠ 预测准确性: 一般 - 仅{metadata['metrics']['within_1']*100:.0f}%的预测在±1以内"
    )

# 高分发现
if discovery_score > 0.5:
    conclusions.append(
        f"✓ 高分发现: 优秀 - 发现了{discovery_score*100:.0f}%的top 10%设计"
    )
else:
    conclusions.append(
        f"⚠ 高分发现: 不足 - 仅发现了{discovery_score*100:.0f}%的top 10%设计"
    )

# 变量覆盖
all_vars_covered = all(
    [
        metadata["coverage"][var]["coverage"] >= 1.0
        for var in ["color_scheme", "layout", "animation"]
    ]
)
if all_vars_covered:
    conclusions.append("✓ 变量覆盖: 完整 - 所有变量的所有水平都被采样")
else:
    conclusions.append("⚠ 变量覆盖: 不完整 - 部分变量水平未被采样")

for conclusion in conclusions:
    print(conclusion)

print()
print("总结:")
if overall_score >= 0.7:
    print("🎉 采样质量优秀！采样数据能够较好地代表完整设计空间。")
    print("   可以基于这些数据进行可靠的统计推断和建模。")
elif overall_score >= 0.5:
    print("✓ 采样质量良好。采样数据基本能够代表设计空间的主要特征。")
    print("  建议在关键区域（如高分区）增加采样以提高代表性。")
else:
    print("⚠ 采样质量一般。采样数据可能无法完全代表设计空间。")
    print("  建议增加采样数量或改进采样策略。")

print("\n" + "=" * 80)
print("分析完成！详细可视化报告将保存到 report/ 目录。")
print("=" * 80)

# 保存详细数据
output_dir = Path(__file__).parent
full_space_df.to_csv(output_dir / "full_design_space_analysis.csv", index=False)
print(f"\n✓ 完整设计空间数据已保存: full_design_space_analysis.csv")

# 保存统计结果
stats_results = {
    "coverage": {
        "sampled_designs": int(n_sampled),
        "total_designs": 360,
        "coverage_rate": float(n_sampled / 360),
    },
    "distribution": {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_pvalue": float(u_pvalue),
    },
    "errors": {
        "mean_error": float(errors.mean()),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "std": float(errors.std()),
    },
    "information": {
        "entropy_full": float(entropy_full),
        "entropy_sampled": float(entropy_sampled),
        "cv_full": float(cv_full),
        "cv_sampled": float(cv_sampled),
    },
    "scores": {
        "coverage_score": float(coverage_score),
        "distribution_score": float(distribution_score),
        "error_score": float(error_score),
        "correlation_score": float(correlation_score),
        "discovery_score": float(discovery_score),
        "overall_score": float(overall_score),
    },
}

with open(output_dir / "statistical_results.json", "w") as f:
    json.dump(stats_results, f, indent=2)
print(f"✓ 统计结果已保存: statistical_results.json")

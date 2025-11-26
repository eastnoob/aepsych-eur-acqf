#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合扰动策略对比测试框架

用途：
- 对比基线配置（随机采样）vs 优化配置（穷举扰动）
- 量化混合扰动策略在2-3水平离散变量场景下的效果

评估指标：
1. 效应发现能力：R²、显著性检测准确率
2. 参数估计质量：RMSE、95% CI覆盖率和宽度
3. 选点质量：采集函数分量分析

作者：eastnoob Code
日期：2025-11-26
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from scipy import stats

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "temp_aepsych"))
sys.path.insert(0, str(PROJECT_ROOT / "extensions" / "dynamic_eur_acquisition"))


def generate_synthetic_data_2_3_levels(
    n_variables: int = 6,
    n_candidates: int = 100,
    true_main_effects: Optional[List[float]] = None,
    true_interactions: Optional[Dict[Tuple[int, int], float]] = None,
    noise_std: float = 0.1,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    生成模拟数据（6个变量，每个2-3水平）

    Args:
        n_variables: 变量数量
        n_candidates: 候选点数量
        true_main_effects: 真实主效应强度（如果None则随机生成）
        true_interactions: 真实交互效应字典 {(i,j): strength}
        noise_std: 观测噪声标准差
        random_seed: 随机种子

    Returns:
        X_pool: (n_candidates, n_variables) 候选点池
        y_true_fn: 真实响应函数（用于生成y）
        ground_truth: 包含真实效应的字典
    """
    rng = np.random.default_rng(random_seed)

    # 生成候选池（2-3水平的离散变量）
    # 前3个变量：2水平（0,1）
    # 后3个变量：3水平（0,1,2）
    X_pool = []
    for i in range(n_candidates):
        x = []
        for j in range(n_variables):
            if j < 3:
                # 2水平变量
                x.append(rng.choice([0, 1]))
            else:
                # 3水平变量
                x.append(rng.choice([0, 1, 2]))
        X_pool.append(x)

    X_pool = np.array(X_pool, dtype=float)

    # 生成真实效应
    if true_main_effects is None:
        # 随机生成（稀疏效应：只有部分维度有显著效应）
        true_main_effects = []
        for i in range(n_variables):
            if rng.random() < 0.5:  # 50%概率有显著效应
                effect = rng.uniform(0.3, 1.0)  # 中等到强效应
            else:
                effect = rng.uniform(0.0, 0.1)  # 弱效应或无效应
            true_main_effects.append(effect)

    if true_interactions is None:
        # 随机生成交互效应（更稀疏：10-20%概率）
        true_interactions = {}
        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                if rng.random() < 0.15:  # 15%概率有显著交互
                    strength = rng.uniform(0.2, 0.8)
                    true_interactions[(i, j)] = strength

    # 定义真实响应函数
    def y_true_fn(X: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """计算真实响应值"""
        n = X.shape[0]
        y = np.zeros(n)

        # 主效应
        for i, coef in enumerate(true_main_effects):
            y += coef * X[:, i]

        # 交互效应
        for (i, j), coef in true_interactions.items():
            y += coef * X[:, i] * X[:, j]

        # 添加噪声
        if add_noise:
            y += rng.normal(0, noise_std, size=n)

        return y

    # 存储真实效应信息
    ground_truth = {
        "main_effects": true_main_effects,
        "interactions": true_interactions,
        "noise_std": noise_std,
        "n_variables": n_variables,
    }

    return X_pool, y_true_fn, ground_truth


def run_single_experiment(
    config_path: str,
    X_pool: np.ndarray,
    y_true_fn: callable,
    n_warmup: int = 10,
    n_eur: int = 15,
    random_seed: int = 42,
) -> Dict:
    """
    运行单次实验

    Args:
        config_path: 配置文件路径
        X_pool: 候选点池
        y_true_fn: 真实响应函数
        n_warmup: warmup阶段样本数
        n_eur: EUR阶段样本数
        random_seed: 随机种子

    Returns:
        results: 包含实验结果的字典
    """
    # TODO: 实现完整的实验流程
    # 1. 加载配置
    # 2. 创建服务器
    # 3. Warmup阶段
    # 4. EUR阶段
    # 5. 评估效应发现能力
    # 6. 返回结果

    results = {
        "config": config_path,
        "seed": random_seed,
        "n_train": n_warmup + n_eur,
        # 评估指标（待实现）
        "R2": None,
        "RMSE": None,
        "effect_detection_accuracy": None,
        "selected_points": None,
    }

    warnings.warn("run_single_experiment is not fully implemented yet")
    return results


def compute_effect_detection_metrics(
    model,
    ground_truth: Dict,
    significance_level: float = 0.05
) -> Dict:
    """
    计算效应发现的准确率

    Args:
        model: 拟合的模型
        ground_truth: 真实效应字典
        significance_level: 显著性水平

    Returns:
        metrics: 包含准确率、召回率、F1分数的字典
    """
    # TODO: 实现效应检测评估
    # 1. 从模型提取系数和p值
    # 2. 与真实效应对比
    # 3. 计算TP, FP, FN, TN
    # 4. 计算准确率、召回率、F1

    metrics = {
        "main_effect_precision": None,
        "main_effect_recall": None,
        "main_effect_f1": None,
        "interaction_precision": None,
        "interaction_recall": None,
        "interaction_f1": None,
    }

    warnings.warn("compute_effect_detection_metrics is not fully implemented yet")
    return metrics


def run_comparison_experiment(
    baseline_config: str,
    optimized_config: str,
    n_repeats: int = 10,
    n_warmup: int = 10,
    n_eur: int = 15,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    运行对比实验（多次重复）

    Args:
        baseline_config: 基线配置文件路径
        optimized_config: 优化配置文件路径
        n_repeats: 重复次数
        n_warmup: warmup阶段样本数
        n_eur: EUR阶段样本数
        output_dir: 结果保存目录

    Returns:
        results_df: 包含所有实验结果的DataFrame
    """
    results = []

    # 生成共享的模拟数据（所有实验使用相同的真实效应）
    X_pool, y_true_fn, ground_truth = generate_synthetic_data_2_3_levels(
        n_variables=6,
        n_candidates=200,
        random_seed=42,
    )

    print("=" * 70)
    print("混合扰动策略对比实验")
    print("=" * 70)
    print(f"\n实验设置：")
    print(f"  - 变量数: {ground_truth['n_variables']}")
    print(f"  - 候选池大小: {len(X_pool)}")
    print(f"  - Warmup样本数: {n_warmup}")
    print(f"  - EUR样本数: {n_eur}")
    print(f"  - 总样本数: {n_warmup + n_eur}")
    print(f"  - 重复次数: {n_repeats}")
    print(f"  - 噪声标准差: {ground_truth['noise_std']}")

    print(f"\n真实效应：")
    print(f"  - 主效应: {ground_truth['main_effects']}")
    print(f"  - 交互效应数量: {len(ground_truth['interactions'])}")

    # 运行实验
    for repeat_idx in range(n_repeats):
        print(f"\n{'='*70}")
        print(f"重复 {repeat_idx + 1}/{n_repeats}")
        print(f"{'='*70}")

        # 基线组
        print(f"\n运行基线配置 (seed={42 + repeat_idx})...")
        baseline_result = run_single_experiment(
            config_path=baseline_config,
            X_pool=X_pool,
            y_true_fn=y_true_fn,
            n_warmup=n_warmup,
            n_eur=n_eur,
            random_seed=42 + repeat_idx,
        )
        baseline_result["group"] = "baseline"
        baseline_result["repeat"] = repeat_idx
        results.append(baseline_result)

        # 优化组
        print(f"运行优化配置 (seed={42 + repeat_idx})...")
        optimized_result = run_single_experiment(
            config_path=optimized_config,
            X_pool=X_pool,
            y_true_fn=y_true_fn,
            n_warmup=n_warmup,
            n_eur=n_eur,
            random_seed=42 + repeat_idx,
        )
        optimized_result["group"] = "optimized"
        optimized_result["repeat"] = repeat_idx
        results.append(optimized_result)

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 保存结果
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(output_path / f"comparison_results_{timestamp}.csv", index=False)

        # 保存真实效应
        with open(output_path / f"ground_truth_{timestamp}.json", "w") as f:
            # 转换ground_truth为可序列化格式
            gt_serializable = {
                "main_effects": ground_truth["main_effects"],
                "interactions": {f"{i},{j}": v for (i, j), v in ground_truth["interactions"].items()},
                "noise_std": ground_truth["noise_std"],
                "n_variables": ground_truth["n_variables"],
            }
            json.dump(gt_serializable, f, indent=2)

        print(f"\n结果已保存至: {output_path}")

    return results_df, ground_truth


def analyze_comparison_results(results_df: pd.DataFrame) -> Dict:
    """
    分析对比实验结果

    Args:
        results_df: 实验结果DataFrame

    Returns:
        analysis: 包含统计分析结果的字典
    """
    baseline_results = results_df[results_df["group"] == "baseline"]
    optimized_results = results_df[results_df["group"] == "optimized"]

    analysis = {
        "summary": {},
        "statistical_tests": {},
    }

    # TODO: 实现完整的统计分析
    # 1. 描述性统计（均值、标准差）
    # 2. 配对t检验
    # 3. Wilcoxon符号秩检验
    # 4. Cohen's d效应量

    print("\n" + "=" * 70)
    print("统计分析结果")
    print("=" * 70)

    # 汇总统计（示例）
    for metric in ["R2", "RMSE", "effect_detection_accuracy"]:
        if metric in results_df.columns and results_df[metric].notna().any():
            baseline_mean = baseline_results[metric].mean()
            baseline_std = baseline_results[metric].std()
            optimized_mean = optimized_results[metric].mean()
            optimized_std = optimized_results[metric].std()

            print(f"\n{metric}:")
            print(f"  基线组: {baseline_mean:.4f} ± {baseline_std:.4f}")
            print(f"  优化组: {optimized_mean:.4f} ± {optimized_std:.4f}")

            # 配对t检验
            if len(baseline_results) == len(optimized_results):
                t_stat, p_value = stats.ttest_rel(
                    optimized_results[metric].values,
                    baseline_results[metric].values
                )
                print(f"  t检验: t={t_stat:.4f}, p={p_value:.4f}")

                # Cohen's d
                diff = optimized_results[metric].values - baseline_results[metric].values
                cohens_d = diff.mean() / diff.std()
                print(f"  Cohen's d: {cohens_d:.4f}")

                analysis["statistical_tests"][metric] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "cohens_d": cohens_d,
                }

    warnings.warn("analyze_comparison_results is not fully implemented yet")
    return analysis


def main():
    """主函数：运行完整的对比实验"""
    # 配置文件路径
    CONFIG_DIR = Path(__file__).parent.parent / "configs"
    baseline_config = str(CONFIG_DIR / "baseline_config.ini")
    optimized_config = str(CONFIG_DIR / "hybrid_perturbation_optimized.ini")

    # 输出目录
    OUTPUT_DIR = Path(__file__).parent.parent / "results" / "hybrid_perturbation_comparison"

    # 运行对比实验
    results_df, ground_truth = run_comparison_experiment(
        baseline_config=baseline_config,
        optimized_config=optimized_config,
        n_repeats=10,  # 10次重复
        n_warmup=10,
        n_eur=15,
        output_dir=str(OUTPUT_DIR),
    )

    # 分析结果
    analysis = analyze_comparison_results(results_df)

    # 保存分析结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"analysis_{timestamp}.json", "w") as f:
        # 转换为可序列化格式
        analysis_serializable = {}
        for key, value in analysis.items():
            if isinstance(value, dict):
                analysis_serializable[key] = {
                    k: (float(v) if isinstance(v, (np.float32, np.float64)) else v)
                    for k, v in value.items()
                }
            else:
                analysis_serializable[key] = value
        json.dump(analysis_serializable, f, indent=2)

    print(f"\n分析结果已保存至: {OUTPUT_DIR / f'analysis_{timestamp}.json'}")

    return results_df, analysis


if __name__ == "__main__":
    # 运行对比实验
    results_df, analysis = main()

    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)
    print("\n下一步：")
    print("1. 完善 run_single_experiment() 函数（实现完整的采样流程）")
    print("2. 完善 compute_effect_detection_metrics() 函数（效应检测评估）")
    print("3. 可视化结果（使用 matplotlib 或 seaborn）")
    print("4. 撰写实验报告")

"""
V3采集函数简化测试：直接对比V1、V2、V3A、V3C的性能

不依赖完整的AEPsych框架，使用简化的采样循环来快速验证效果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from virtual_user import VirtualUser

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 导入V1和V3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from acquisition_function import VarianceReductionWithCoverageAcqf
from acquisition_function_v3 import HardExclusionAcqf, CombinedAcqf


def generate_all_designs():
    """生成所有可能的设计（360个）"""
    designs = []
    # color: 5, layout: 4, font_size: 6, background: 3
    for c in range(5):
        for l in range(4):
            for f in range(6):
                for b in range(3):
                    designs.append([c, l, f, b])
    return np.array(designs)


def run_simplified_experiment(acqf_class, acqf_name, n_trials=80, n_init=20, seed=42):
    """
    运行简化实验
    
    Args:
        acqf_class: 采集函数类
        acqf_name: 采集函数名称
        n_trials: 总试验次数
        n_init: 初始随机采样数
        seed: 随机种子
    """
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"运行 {acqf_name} 实验")
    print(f"{'='*70}")
    
    # 虚拟用户
    virtual_user = VirtualUser(
        main_effects={
            'color': [0.0, 1.5, 0.5, -0.5, -1.0],
            'layout': [0.0, 1.0, 0.3, -0.8],
            'font_size': [0.0, 0.2, 0.5, 0.8, 1.0, 0.6],
            'background': [0.0, -0.5, 0.3]
        },
        interactions={
            (0, 1): 1.0,
            (1, 3): -0.8,
            (2, 3): 0.6
        },
        noise_std=0.5,
        seed=seed
    )
    
    # 生成所有设计
    all_designs = generate_all_designs()
    
    # 初始化采集函数
    acqf = acqf_class(
        model=None,
        lambda_min=0.5,
        lambda_max=3.0,
        tau_1=0.5,
        tau_2=0.3,
        gamma=0.5,
        interaction_terms=[(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    )
    
    # 存储结果
    X_sampled = []
    y_sampled = []
    true_scores = []
    
    # 第一阶段：随机初始化
    print(f"阶段1: 随机初始化 ({n_init} 次)")
    init_indices = np.random.choice(len(all_designs), size=n_init, replace=False)
    for idx in init_indices:
        x = all_designs[idx]
        response = virtual_user.respond(x)
        true_score = virtual_user.get_true_score(x)
        
        X_sampled.append(x)
        y_sampled.append(response)
        true_scores.append(true_score)
    
    X_sampled = np.array(X_sampled)
    y_sampled = np.array(y_sampled)
    
    # 第二阶段：采集函数引导
    print(f"阶段2: 采集函数引导 ({n_trials - n_init} 次)")
    for trial in range(n_init + 1, n_trials + 1):
        # 拟合模型
        acqf.fit(X_sampled, y_sampled)
        
        # 评估所有候选
        if isinstance(acqf, CombinedAcqf):
            # 组合方案：预过滤候选集
            candidates = acqf.filter_candidates(all_designs)
        else:
            # 其他方案：使用所有设计
            candidates = all_designs
        
        # 评分
        scores = acqf._evaluate_numpy(candidates)
        
        # 选择最佳
        best_idx = np.argmax(scores)
        x_next = candidates[best_idx]
        
        # 虚拟用户响应
        response = virtual_user.respond(x_next)
        true_score = virtual_user.get_true_score(x_next)
        
        # 记录
        X_sampled = np.vstack([X_sampled, x_next])
        y_sampled = np.append(y_sampled, response)
        true_scores.append(true_score)
        
        if trial % 10 == 0:
            print(f"  Trial {trial}/{n_trials}: 得分 {true_score:.2f}")
    
    # 计算指标
    results = calculate_metrics(X_sampled, y_sampled, true_scores, acqf_name, all_designs)
    
    return results


def calculate_metrics(X_sampled, y_sampled, true_scores, acqf_name, all_designs):
    """计算评估指标"""
    
    # 1. 空间覆盖指标
    unique_designs = set()
    for x in X_sampled:
        design_str = "_".join([f"{int(v)}" for v in x])
        unique_designs.add(design_str)
    
    n_unique = len(unique_designs)
    n_total = len(all_designs)
    coverage_rate = n_unique / n_total
    repeat_rate = (len(X_sampled) - n_unique) / len(X_sampled)
    
    # 2. 质量发现指标
    true_scores_arr = np.array(true_scores)
    high_score_counts = {
        'score_ge_9.0': int(np.sum(true_scores_arr >= 9.0)),
        'score_ge_9.5': int(np.sum(true_scores_arr >= 9.5)),
        'score_ge_10.0': int(np.sum(true_scores_arr >= 10.0))
    }
    
    # 3. 因子水平覆盖
    level_coverage = {}
    factor_names = ['color', 'layout', 'font_size', 'background']
    factor_levels = [5, 4, 6, 3]
    
    for i, (name, n_levels) in enumerate(zip(factor_names, factor_levels)):
        sampled_levels = set(X_sampled[:, i].astype(int))
        level_coverage[name] = {
            'sampled': len(sampled_levels),
            'total': n_levels,
            'rate': len(sampled_levels) / n_levels
        }
    
    # 4. 关键交互项覆盖
    key_interactions = [(0, 1), (1, 3), (2, 3)]
    interaction_coverage = {}
    
    for i, j in key_interactions:
        sampled_pairs = set()
        for x in X_sampled:
            pair = (int(x[i]), int(x[j]))
            sampled_pairs.add(pair)
        
        total_pairs = factor_levels[i] * factor_levels[j]
        interaction_coverage[f"{factor_names[i]}_{factor_names[j]}"] = {
            'sampled_pairs': len(sampled_pairs),
            'total_pairs': total_pairs,
            'rate': len(sampled_pairs) / total_pairs
        }
    
    results = {
        'acqf_name': acqf_name,
        'n_trials': len(X_sampled),
        'space_coverage': {
            'unique_designs': n_unique,
            'total_designs': n_total,
            'coverage_rate': float(coverage_rate),
            'repeat_rate': float(repeat_rate),
            'level_coverage': level_coverage,
            'interaction_coverage': interaction_coverage
        },
        'quality_discovery': {
            'mean_score': float(np.mean(true_scores_arr)),
            'std_score': float(np.std(true_scores_arr)),
            'min_score': float(np.min(true_scores_arr)),
            'max_score': float(np.max(true_scores_arr)),
            'high_score_counts': high_score_counts
        },
        'trial_data': {
            'X': X_sampled.tolist(),
            'y': y_sampled.tolist(),
            'true_scores': true_scores
        }
    }
    
    return results


def print_comparison(results_dict):
    """打印对比结果"""
    print(f"\n{'='*70}")
    print("实验结果对比")
    print(f"{'='*70}\n")
    
    # 表头
    names = list(results_dict.keys())
    print(f"{'指标':<30} " + " ".join([f"{name:>12}" for name in names]))
    print("-" * 70)
    
    # 空间覆盖
    print("\n【空间覆盖】")
    row = "唯一设计数"
    values = [results_dict[name]['space_coverage']['unique_designs'] for name in names]
    print(f"{row:<30} " + " ".join([f"{v:>12}" for v in values]))
    
    row = "覆盖率 (%)"
    values = [results_dict[name]['space_coverage']['coverage_rate']*100 for name in names]
    print(f"{row:<30} " + " ".join([f"{v:>12.1f}" for v in values]))
    
    row = "重复率 (%)"
    values = [results_dict[name]['space_coverage']['repeat_rate']*100 for name in names]
    print(f"{row:<30} " + " ".join([f"{v:>12.1f}" for v in values]))
    
    # 质量发现
    print("\n【质量发现】")
    row = "平均分数"
    values = [results_dict[name]['quality_discovery']['mean_score'] for name in names]
    print(f"{row:<30} " + " ".join([f"{v:>12.2f}" for v in values]))
    
    row = "标准差"
    values = [results_dict[name]['quality_discovery']['std_score'] for name in names]
    print(f"{row:<30} " + " ".join([f"{v:>12.2f}" for v in values]))
    
    row = "最高分"
    values = [results_dict[name]['quality_discovery']['max_score'] for name in names]
    print(f"{row:<30} " + " ".join([f"{v:>12.2f}" for v in values]))
    
    row = "高分发现 (≥9.5)"
    values = [results_dict[name]['quality_discovery']['high_score_counts']['score_ge_9.5'] for name in names]
    print(f"{row:<30} " + " ".join([f"{v:>12}" for v in values]))
    
    # 因子覆盖平均
    print("\n【平均因子水平覆盖率】")
    for name in names:
        rates = [cov['rate'] for cov in results_dict[name]['space_coverage']['level_coverage'].values()]
        avg_rate = np.mean(rates) * 100
        print(f"  {name}: {avg_rate:.1f}%")
    
    # 交互项覆盖平均
    print("\n【平均交互项覆盖率】")
    for name in names:
        rates = [cov['rate'] for cov in results_dict[name]['space_coverage']['interaction_coverage'].values()]
        avg_rate = np.mean(rates) * 100
        print(f"  {name}: {avg_rate:.1f}%")


def save_results(results_dict, output_dir="results_v3_comparison"):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存JSON
    json_path = os.path.join(output_dir, f"comparison_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存: {json_path}")
    
    return json_path


if __name__ == "__main__":
    print("="*70)
    print("V3采集函数对比实验")
    print("="*70)
    print("\n对比方案:")
    print("  V1: 基线 (VarianceReductionWithCoverageAcqf)")
    print("  V3A: V1 + 硬排除 (HardExclusionAcqf)")
    print("  V3C: V1 + 候选集过滤 + 硬排除 (CombinedAcqf)")
    print()
    
    # 运行实验
    results_v1 = run_simplified_experiment(
        VarianceReductionWithCoverageAcqf, "V1-Baseline",
        n_trials=80, n_init=20, seed=42
    )
    
    results_v3a = run_simplified_experiment(
        HardExclusionAcqf, "V3A-HardExclusion",
        n_trials=80, n_init=20, seed=42
    )
    
    results_v3c = run_simplified_experiment(
        CombinedAcqf, "V3C-Combined",
        n_trials=80, n_init=20, seed=42
    )
    
    # 整合结果
    results_dict = {
        'V1': results_v1,
        'V3A': results_v3a,
        'V3C': results_v3c
    }
    
    # 打印对比
    print_comparison(results_dict)
    
    # 保存
    save_results(results_dict)
    
    print(f"\n{'='*70}")
    print("✅ V3对比实验完成")
    print(f"{'='*70}")

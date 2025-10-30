"""
V3实验运行脚本：测试方案A(硬排除)和方案C(组合方案)

评估指标设计基于用户目标：
- 目标: 从有限试验中最大化主效应和交互效应估计精度
- 实验类型: 分类自变量 -> Likert量表因变量
- 关注: 主效应 + 二阶交互效应

评估维度:
1. 效应估计精度 (核心)
   - 参数方差 (不确定性)
   - 参数覆盖 (估计了哪些效应)
   - 模型拟合度 (R², MSE)

2. 空间探索效率
   - 唯一设计数
   - 因子水平覆盖
   - 交互项设计覆盖

3. 高质量发现
   - 高分设计数量
   - 分数分布
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 导入必要模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from acquisition_function_v3 import HardExclusionAcqf, CombinedAcqf

sys.path.insert(0, os.path.dirname(__file__))
from virtual_user import BalancedVirtualUser

# AEPsych imports
from aepsych.config import Config
from aepsych.server import AEPsychServer
from aepsych.models import GPClassificationModel
from aepsych.generators import OptimizeAcqfGenerator, SobolGenerator


class V3ExperimentRunner:
    """V3实验运行器"""

    def __init__(self, config_path, acqf_type="hard_exclusion", n_trials=80, seed=42):
        """
        Args:
            config_path: 配置文件路径
            acqf_type: "hard_exclusion" (方案A) 或 "combined" (方案C)
            n_trials: 试验次数
            seed: 随机种子
        """
        self.config_path = config_path
        self.acqf_type = acqf_type
        self.n_trials = n_trials
        self.seed = seed

        np.random.seed(seed)

        # 虚拟用户 (与V1/V2相同)
        self.virtual_user = BalancedVirtualUser(
            main_effects={
                "color": [0.0, 1.5, 0.5, -0.5, -1.0],
                "layout": [0.0, 1.0, 0.3, -0.8],
                "font_size": [0.0, 0.2, 0.5, 0.8, 1.0, 0.6],
                "background": [0.0, -0.5, 0.3],
            },
            interactions={
                (0, 1): 1.0,  # color x layout
                (1, 3): -0.8,  # layout x background
                (2, 3): 0.6,  # font_size x background
            },
            noise_std=0.5,
            seed=seed,
        )

        # 因子信息
        self.factor_names = ["color", "layout", "font_size", "background"]
        self.factor_levels = {
            "color": ["red", "orange", "yellow", "green", "blue"],
            "layout": ["list", "grid", "table", "card"],
            "font_size": [
                "Font=10",
                "Font=11",
                "Font=12",
                "Font=13",
                "Font=14",
                "Font=15",
            ],
            "background": ["white", "gray", "none"],
        }

        # 存储结果
        self.trial_data = []
        self.metadata = {}

    def setup_server(self):
        """设置AEPsych服务器"""
        # 注册自定义采集函数
        if self.acqf_type == "hard_exclusion":
            OptimizeAcqfGenerator.register_acqf(HardExclusionAcqf, "HardExclusionAcqf")
        elif self.acqf_type == "combined":
            OptimizeAcqfGenerator.register_acqf(CombinedAcqf, "CombinedAcqf")

        # 加载配置
        with open(self.config_path, "r") as f:
            config_str = f.read()

        config = Config(config_str=config_str)
        self.server = AEPsychServer(config=config)

    def run_trial(self, trial_num):
        """运行单次试验"""
        # 询问下一个试验配置
        trial_config = self.server.ask()

        # 解析配置
        config_dict = {}
        for i, param_name in enumerate(self.factor_names):
            param_value = trial_config["config"][i]
            levels = self.factor_levels[param_name]
            config_dict[param_name] = levels[int(param_value)]

        # 虚拟用户响应
        response = self.virtual_user.respond(trial_config["config"])
        true_score = self.virtual_user.get_true_score(trial_config["config"])

        # 告知服务器结果
        self.server.tell(config=trial_config, outcome=response)

        # 记录数据
        trial_record = {
            "trial": trial_num,
            **config_dict,
            "response": response,
            "true_score": true_score,
            "config_vector": trial_config["config"].tolist(),
        }
        self.trial_data.append(trial_record)

        if trial_num % 10 == 0:
            print(
                f"  Trial {trial_num}/{self.n_trials}: {config_dict} -> {response:.3f} (true: {true_score:.2f})"
            )

        return trial_record

    def run_experiment(self):
        """运行完整实验"""
        print(f"\n{'='*70}")
        print(f"运行V3实验: {self.acqf_type}")
        print(f"{'='*70}")
        print(f"配置: {self.config_path}")
        print(f"试验数: {self.n_trials}")
        print(f"随机种子: {self.seed}")
        print()

        self.setup_server()

        for trial_num in range(1, self.n_trials + 1):
            self.run_trial(trial_num)

        print(f"\n✓ 实验完成")

    def calculate_effect_estimation_metrics(self):
        """
        计算效应估计精度指标（核心评估维度）

        基于用户目标：最大化主效应和交互效应估计精度
        """
        print("\n计算效应估计精度指标...")

        model = self.server.strat.model

        # 1. 参数不确定性 (方差)
        # 获取模型后验方差
        X_all = np.array([d["config_vector"] for d in self.trial_data])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = model.predict(torch.tensor(X_all, dtype=torch.float32))

        # 平均预测方差 (越小越好)
        mean_variance = predictions[1].mean().item() if len(predictions) > 1 else 0.0

        # 2. 模型拟合度
        y_true = np.array([d["response"] for d in self.trial_data])
        y_pred = predictions[0].detach().numpy().flatten()

        # R² (越接近1越好)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # MSE (越小越好)
        mse = np.mean((y_true - y_pred) ** 2)

        # 3. 真实效应恢复度 (如果已知真实效应)
        # 由于我们有虚拟用户，可以评估估计准确度
        true_scores = np.array([d["true_score"] for d in self.trial_data])

        # 真实分数的预测MSE
        true_score_mse = np.mean((true_scores - y_pred) ** 2)

        # 真实分数R²
        ss_res_true = np.sum((true_scores - y_pred) ** 2)
        ss_tot_true = np.sum((true_scores - true_scores.mean()) ** 2)
        true_r_squared = 1 - (ss_res_true / ss_tot_true) if ss_tot_true > 0 else 0.0

        metrics = {
            "mean_prediction_variance": float(mean_variance),
            "observed_r_squared": float(r_squared),
            "observed_mse": float(mse),
            "true_score_mse": float(true_score_mse),
            "true_score_r_squared": float(true_r_squared),
        }

        return metrics

    def calculate_space_coverage_metrics(self):
        """计算空间覆盖指标"""
        print("计算空间覆盖指标...")

        # 唯一设计
        unique_designs = set()
        for d in self.trial_data:
            design_str = "_".join(
                [d["color"], d["layout"], d["font_size"], d["background"]]
            )
            unique_designs.add(design_str)

        # 总设计空间
        total_designs = (
            len(self.factor_levels["color"])
            * len(self.factor_levels["layout"])
            * len(self.factor_levels["font_size"])
            * len(self.factor_levels["background"])
        )

        # 因子水平覆盖
        level_coverage = {}
        for factor in self.factor_names:
            sampled_levels = set([d[factor] for d in self.trial_data])
            level_coverage[factor] = {
                "sampled": len(sampled_levels),
                "total": len(self.factor_levels[factor]),
                "coverage_rate": len(sampled_levels) / len(self.factor_levels[factor]),
            }

        # 交互项设计覆盖 (关键交互对的水平组合覆盖)
        key_interactions = [
            (0, 1),
            (1, 3),
            (2, 3),
        ]  # color-layout, layout-background, fontsize-background
        interaction_coverage = {}

        for i, j in key_interactions:
            factor_i = self.factor_names[i]
            factor_j = self.factor_names[j]

            sampled_pairs = set()
            for d in self.trial_data:
                pair = (d[factor_i], d[factor_j])
                sampled_pairs.add(pair)

            total_pairs = len(self.factor_levels[factor_i]) * len(
                self.factor_levels[factor_j]
            )

            interaction_coverage[f"{factor_i}_{factor_j}"] = {
                "sampled_pairs": len(sampled_pairs),
                "total_pairs": total_pairs,
                "coverage_rate": len(sampled_pairs) / total_pairs,
            }

        metrics = {
            "unique_designs": len(unique_designs),
            "total_designs": total_designs,
            "coverage_rate": len(unique_designs) / total_designs,
            "repeat_rate": (len(self.trial_data) - len(unique_designs))
            / len(self.trial_data),
            "level_coverage": level_coverage,
            "interaction_coverage": interaction_coverage,
        }

        return metrics

    def calculate_quality_discovery_metrics(self):
        """计算高质量发现指标"""
        print("计算高质量发现指标...")

        true_scores = [d["true_score"] for d in self.trial_data]

        # 高分阈值
        thresholds = [9.0, 9.5, 10.0]
        high_score_counts = {}
        for thresh in thresholds:
            count = sum(1 for score in true_scores if score >= thresh)
            high_score_counts[f"score_ge_{thresh}"] = count

        # 分数分布
        metrics = {
            "mean_score": float(np.mean(true_scores)),
            "std_score": float(np.std(true_scores)),
            "min_score": float(np.min(true_scores)),
            "max_score": float(np.max(true_scores)),
            "high_score_counts": high_score_counts,
        }

        return metrics

    def analyze_results(self):
        """综合分析结果"""
        print(f"\n{'='*70}")
        print("结果分析")
        print(f"{'='*70}")

        # 1. 效应估计精度 (核心)
        effect_metrics = self.calculate_effect_estimation_metrics()

        # 2. 空间覆盖
        coverage_metrics = self.calculate_space_coverage_metrics()

        # 3. 高质量发现
        quality_metrics = self.calculate_quality_discovery_metrics()

        # 整合
        self.metadata = {
            "acqf_type": self.acqf_type,
            "config_path": self.config_path,
            "n_trials": self.n_trials,
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            "effect_estimation": effect_metrics,
            "space_coverage": coverage_metrics,
            "quality_discovery": quality_metrics,
        }

        # 打印摘要
        print("\n【效应估计精度】(核心指标)")
        print(f"  平均预测方差: {effect_metrics['mean_prediction_variance']:.4f}")
        print(f"  观测数据R²: {effect_metrics['observed_r_squared']:.3f}")
        print(f"  观测数据MSE: {effect_metrics['observed_mse']:.4f}")
        print(f"  真实分数R²: {effect_metrics['true_score_r_squared']:.3f}")
        print(f"  真实分数MSE: {effect_metrics['true_score_mse']:.4f}")

        print("\n【空间覆盖】")
        print(
            f"  唯一设计: {coverage_metrics['unique_designs']}/{coverage_metrics['total_designs']} ({coverage_metrics['coverage_rate']*100:.1f}%)"
        )
        print(f"  重复率: {coverage_metrics['repeat_rate']*100:.1f}%")

        print("\n  因子水平覆盖:")
        for factor, cov in coverage_metrics["level_coverage"].items():
            print(
                f"    {factor}: {cov['sampled']}/{cov['total']} ({cov['coverage_rate']*100:.1f}%)"
            )

        print("\n  关键交互项覆盖:")
        for inter, cov in coverage_metrics["interaction_coverage"].items():
            print(
                f"    {inter}: {cov['sampled_pairs']}/{cov['total_pairs']} ({cov['coverage_rate']*100:.1f}%)"
            )

        print("\n【高质量发现】")
        print(
            f"  平均真实分数: {quality_metrics['mean_score']:.2f} ± {quality_metrics['std_score']:.2f}"
        )
        print(
            f"  分数范围: [{quality_metrics['min_score']:.2f}, {quality_metrics['max_score']:.2f}]"
        )
        print(f"  高分发现:")
        for thresh, count in quality_metrics["high_score_counts"].items():
            print(f"    {thresh.replace('_', ' ')}: {count}个")

        return self.metadata

    def save_results(self, output_dir="results_v3"):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.acqf_type}_{timestamp}"

        # 保存试验数据
        df = pd.DataFrame(self.trial_data)
        csv_path = os.path.join(output_dir, f"trial_data_{prefix}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ 试验数据已保存: {csv_path}")

        # 保存元数据
        meta_path = os.path.join(output_dir, f"metadata_{prefix}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ 元数据已保存: {meta_path}")

        return csv_path, meta_path


import torch


def run_v3_experiment(acqf_type, config_suffix):
    """
    运行V3实验

    Args:
        acqf_type: "hard_exclusion" 或 "combined"
        config_suffix: "v3a" 或 "v3c"
    """
    config_path = f"experiment_config_{config_suffix}.ini"

    runner = V3ExperimentRunner(
        config_path=config_path, acqf_type=acqf_type, n_trials=80, seed=42
    )

    runner.run_experiment()
    metadata = runner.analyze_results()
    runner.save_results()

    return runner, metadata


if __name__ == "__main__":
    print("=" * 70)
    print("V3采集函数实验")
    print("=" * 70)
    print("\n测试方案:")
    print("  方案A: V1 + 硬排除")
    print("  方案C: V1 + 候选集过滤 + 硬排除")
    print()

    # 方案A
    print("\n" + "=" * 70)
    print("实验1: 方案A (硬排除)")
    print("=" * 70)
    runner_a, meta_a = run_v3_experiment("hard_exclusion", "v3a")

    # 方案C
    print("\n" + "=" * 70)
    print("实验2: 方案C (组合方案)")
    print("=" * 70)
    runner_c, meta_c = run_v3_experiment("combined", "v3c")

    print("\n" + "=" * 70)
    print("✅ V3实验完成")
    print("=" * 70)

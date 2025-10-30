#!/usr/bin/env python3
"""
V1 vs V3A 对比实验
测试纯硬排除逻辑是否能达到 0% 重复率
"""

import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from virtual_user import VirtualUser

# 导入AEPsych组件
try:
    from aepsych.server.server import AEPsychServer
    from aepsych.server.message_handlers.handle_setup import configure
    from aepsych.server.message_handlers.handle_ask import ask
    from aepsych.server.message_handlers.handle_tell import tell
    from aepsych.config import Config

    # 注册自定义采集函数模块
    sys.path.insert(0, str(project_root / "extensions"))
    import dynamic_eur_acquisition

    Config.register_module(dynamic_eur_acquisition)

    HAS_AEPSYCH_SERVER = True
except ImportError as e:
    print(f"警告: 无法导入AEPsych Server模块: {e}")
    HAS_AEPSYCH_SERVER = False


class V1V3AComparison:
    """V1 vs V3A 对比实验"""

    def __init__(
        self,
        user_type: str = "balanced",
        user_noise: float = 0.5,
        seed: int = 42,
    ):
        self.user_type = user_type
        self.user_noise = user_noise
        self.seed = seed

        # 创建results目录
        self.results_dir = Path(__file__).parent / "results_v1_v3a"
        self.results_dir.mkdir(exist_ok=True)

        # 实验配置
        self.configs = {
            "V1": "experiment_config.ini",
            "V3A": "experiment_config_v3a.ini",
        }

        # 存储所有实验结果
        self.all_results = {}

        print(f"\n{'='*80}")
        print(f" V1 vs V3A 对比实验 - 硬排除验证")
        print(f"{'='*80}")
        print(f"\n虚拟用户配置:")
        print(f"  - 用户类型: {user_type}")
        print(f"  - 噪声水平: {user_noise}")
        print(f"  - 设计空间: {VirtualUser.design_space_size()} 种组合")
        print(
            f"  - 试验预算: 60 次/方案 ({60/VirtualUser.design_space_size()*100:.1f}% < 1/4)"
        )
        print(f"\n对比方案:")
        print(
            f"  - V1: 基线 (VarianceReductionWithCoverageAcqf + AcqfGridSearchGenerator)"
        )
        print(f"  - V3A: V1 + 硬排除 (HardExclusionAcqf + HardExclusionGenerator)")
        print(f"\n预期结果:")
        print(f"  - V1: ~30/60 唯一设计 (50% 重复率)")
        print(f"  - V3A: 60/60 唯一设计 (0% 重复率) ✅")

    def _design_to_config_values(self, design):
        """将设计字典转换为配置值列表"""
        return [
            design["color_scheme"],
            design["layout"],
            design["font_size"],
            design["animation"],
        ]

    def run_single_experiment(self, method_name: str, config_file: str):
        """运行单个实验"""
        print(f"\n{'='*80}")
        print(f" 运行 {method_name} 实验")
        print(f"{'='*80}")

        # 创建虚拟用户
        user = VirtualUser(
            user_type=self.user_type, noise_level=self.user_noise, seed=self.seed
        )

        # 创建数据库路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = str(self.results_dir / f"experiment_{method_name}_{timestamp}.db")

        # 创建server
        server = AEPsychServer(database_path=db_path)

        # 读取配置
        config_path = Path(__file__).parent / config_file
        with open(config_path, "r", encoding="utf-8") as f:
            config_str = f.read()

        # 创建Config对象并配置server
        config_obj = Config(config_str=config_str)
        configure(server, config=config_obj)

        # 运行实验
        trial_count = 0
        phase = "initialization"
        sampled_designs = []  # 记录所有采样的设计

        # 运行实验循环 (server.strat.finished会在60次后自动停止)
        while not server.strat.finished:
            trial_count += 1

            # Ask - 获取下一个试验点
            config_dict = ask(server)
            if config_dict is None:
                break

            # 转换格式
            config_values = {
                key: val[0] if isinstance(val, list) else val
                for key, val in config_dict.items()
            }
            design = {
                "color_scheme": config_values["color_scheme"],
                "layout": config_values["layout"],
                "font_size": config_values["font_size"],
                "animation": config_values["animation"],
            }

            # 记录设计
            sampled_designs.append(design)

            # 获取用户评分
            result = user.rate_design(design)
            rating = result["rating"]

            # Tell - 告诉server结果
            tell(server, config=config_dict, outcome=rating)

            # 阶段转换
            if trial_count == 15 and phase == "initialization":
                phase = "optimization"
                print(f"\n✓ 初始化阶段完成 (15次试验)")
                print(f"开始优化阶段...")

            # 进度输出
            if trial_count % 10 == 0:
                print(f"  试验 {trial_count}/60 完成...")

        print(f"\n✓ {method_name} 实验完成!")
        print(f"  总试验数: {trial_count}")

        # 分析结果
        results = self._analyze_results(method_name, sampled_designs, user)

        # 数据库已经在创建server时指定路径自动保存
        results["db_path"] = db_path

        return results

    def _analyze_results(self, method_name, sampled_designs, user):
        """分析实验结果"""
        print(f"\n{method_name} 关键指标:")

        # 统计唯一设计
        unique_designs = []
        design_counts = {}

        for design in sampled_designs:
            design_key = tuple(sorted(design.items()))
            if design_key not in design_counts:
                design_counts[design_key] = 0
                unique_designs.append(design)
            design_counts[design_key] += 1

        n_unique = len(unique_designs)
        n_total = len(sampled_designs)
        repeat_rate = (n_total - n_unique) / n_total * 100

        print(f"  唯一设计: {n_unique}/{n_total}")
        print(f"  重复率: {repeat_rate:.1f}%")

        # 计算真实得分
        true_scores = [user.get_ground_truth(d) for d in sampled_designs]
        mean_score = np.mean(true_scores)
        max_score = np.max(true_scores)
        high_score_count = sum(1 for s in true_scores if s >= 9.0)

        print(f"  平均分数: {mean_score:.2f}")
        print(f"  最高分数: {max_score:.2f}")
        print(f"  高分发现: {high_score_count}")

        # ⭐ 关键检查: 如果是 V3A,打印重复设计详情
        if method_name == "V3A" and repeat_rate > 0:
            print(f"\n⚠️ 警告: V3A 仍有重复!")
            print(f"重复设计详情:")
            for design_key, count in design_counts.items():
                if count > 1:
                    design = dict(design_key)
                    print(f"  {design} - 重复 {count} 次")

        return {
            "method": method_name,
            "config": self.configs[method_name],
            "n_trials": n_total,
            "unique_designs": n_unique,
            "repeat_rate": repeat_rate,
            "mean_score": mean_score,
            "max_score": max_score,
            "high_score_count": high_score_count,
        }

    def run_all_experiments(self):
        """运行所有实验"""
        for method_name, config_file in self.configs.items():
            results = self.run_single_experiment(method_name, config_file)
            self.all_results[method_name] = results

        # 生成对比报告
        self._generate_comparison_report()

    def _generate_comparison_report(self):
        """生成对比报告"""
        print(f"\n{'='*80}")
        print(f" 对比结果")
        print(f"{'='*80}\n")

        # 创建DataFrame
        df = pd.DataFrame(self.all_results).T
        print(df.to_string())

        # 保存CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.results_dir / f"comparison_report_{timestamp}.csv"
        df.to_csv(csv_path)
        print(f"\n✓ 对比报告已保存: {csv_path}")

        # 保存JSON
        json_path = self.results_dir / f"comparison_report_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "user_type": self.user_type,
                    "user_noise": self.user_noise,
                    "seed": self.seed,
                    "results": self.all_results,
                },
                f,
                indent=2,
            )
        print(f"✓ JSON报告已保存: {json_path}")

        # 生成可视化
        self._generate_visualization(timestamp)

    def _generate_visualization(self, timestamp):
        """生成可视化"""
        print(f"\n{'='*80}")
        print(f" 生成可视化")
        print(f"{'='*80}\n")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("V1 vs V3A 对比实验结果", fontsize=16, fontweight="bold")

        methods = list(self.all_results.keys())
        colors = {"V1": "#1f77b4", "V3A": "#2ca02c"}

        # 1. 唯一设计数
        ax = axes[0, 0]
        unique_counts = [self.all_results[m]["unique_designs"] for m in methods]
        bars = ax.bar(
            methods, unique_counts, color=[colors[m] for m in methods], alpha=0.8
        )
        ax.set_ylabel("唯一设计数", fontsize=12)
        ax.set_title("唯一设计数量对比", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 70)
        ax.axhline(y=60, color="red", linestyle="--", label="目标: 60 (无重复)")
        for bar, count in zip(bars, unique_counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{count}",
                ha="center",
                va="bottom",
            )
        ax.legend()

        # 2. 重复率
        ax = axes[0, 1]
        repeat_rates = [self.all_results[m]["repeat_rate"] for m in methods]
        bars = ax.bar(
            methods, repeat_rates, color=[colors[m] for m in methods], alpha=0.8
        )
        ax.set_ylabel("重复率 (%)", fontsize=12)
        ax.set_title("重复率对比", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.axhline(y=0, color="green", linestyle="--", label="目标: 0% (无重复)")
        for bar, rate in zip(bars, repeat_rates):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
            )
        ax.legend()

        # 3. 平均得分
        ax = axes[1, 0]
        mean_scores = [self.all_results[m]["mean_score"] for m in methods]
        bars = ax.bar(
            methods, mean_scores, color=[colors[m] for m in methods], alpha=0.8
        )
        ax.set_ylabel("平均真实得分", fontsize=12)
        ax.set_title("平均得分对比", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 10)
        for bar, score in zip(bars, mean_scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{score:.2f}",
                ha="center",
                va="bottom",
            )

        # 4. 高分发现
        ax = axes[1, 1]
        high_score_counts = [self.all_results[m]["high_score_count"] for m in methods]
        bars = ax.bar(
            methods, high_score_counts, color=[colors[m] for m in methods], alpha=0.8
        )
        ax.set_ylabel("高分设计数 (≥9.0)", fontsize=12)
        ax.set_title("高分发现对比", fontsize=12, fontweight="bold")
        for bar, count in zip(bars, high_score_counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{count}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        # 保存
        viz_path = self.results_dir / f"comparison_visualization_{timestamp}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ 可视化已保存: {viz_path}")


if __name__ == "__main__":
    if not HAS_AEPSYCH_SERVER:
        print("错误: 需要AEPsych Server才能运行实验")
        sys.exit(1)

    # 运行实验
    experiment = V1V3AComparison(user_type="balanced", user_noise=0.5, seed=42)
    experiment.run_all_experiments()

    print(f"\n{'='*80}")
    print(f" ✅ V1 vs V3A 对比实验完成!")
    print(f"{'='*80}")
    print(f"\n结果保存在: {experiment.results_dir}")

"""
V3完整对比实验 - 使用AEPsych服务器和数据库

对比三个方案:
- V1: VarianceReductionWithCoverageAcqf (基线)
- V3A: HardExclusionAcqf (硬排除)
- V3C: CombinedAcqf (组合方案)

使用相同的实验框架、虚拟用户、数据库保存
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
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


class V3ComparisonExperiment:
    """
    V3完整对比实验运行器
    """

    def __init__(
        self,
        user_type: str = "balanced",
        user_noise: float = 0.5,
        seed: int = 42,
    ):
        """
        初始化实验

        Parameters
        ----------
        user_type : str
            虚拟用户类型
        user_noise : float
            用户评分噪声
        seed : int
            随机种子
        """
        self.user_type = user_type
        self.user_noise = user_noise
        self.seed = seed

        # 创建results目录
        self.results_dir = Path(__file__).parent / "results_v3_full"
        self.results_dir.mkdir(exist_ok=True)

        # 实验配置
        self.configs = {
            "V1": "experiment_config.ini",
            "V3A": "experiment_config_v3a.ini",
            "V3C": "experiment_config_v3c.ini",
        }

        # 存储所有实验结果
        self.all_results = {}

        print(f"\n{'='*80}")
        print(f" V3完整对比实验")
        print(f"{'='*80}")
        print(f"\n虚拟用户配置:")
        print(f"  - 用户类型: {user_type}")
        print(f"  - 噪声水平: {user_noise}")
        print(f"  - 设计空间: {VirtualUser.design_space_size()} 种组合")
        print(
            f"  - 试验预算: 60 次/方案 ({60/VirtualUser.design_space_size()*100:.1f}% < 1/4)"
        )
        print(f"\n对比方案:")
        print(f"  - V1: 基线 (VarianceReductionWithCoverageAcqf)")
        print(f"  - V3A: V1 + 硬排除 (HardExclusionAcqf)")
        print(f"  - V3C: V1 + 候选集过滤 + 硬排除 (CombinedAcqf)")

    def _design_to_config_values(self, design: Dict) -> List:
        """将设计字典转换为配置值列表"""
        return [
            design["color_scheme"],
            design["layout"],
            design["font_size"],
            design["animation"],
        ]

    def _config_values_to_design(self, values) -> Dict:
        """将配置值转换为设计字典"""
        if isinstance(values, dict):
            return {
                "color_scheme": values["color_scheme"],
                "layout": values["layout"],
                "font_size": (
                    int(values["font_size"])
                    if not isinstance(values["font_size"], str)
                    else values["font_size"]
                ),
                "animation": values["animation"],
            }
        else:
            return {
                "color_scheme": values[0],
                "layout": values[1],
                "font_size": (
                    int(values[2]) if not isinstance(values[2], str) else values[2]
                ),
                "animation": values[3],
            }

    def run_single_experiment(self, method_name: str, config_path: str) -> Dict:
        """
        运行单个实验方案

        Parameters
        ----------
        method_name : str
            方案名称 (V1, V3A, V3C)
        config_path : str
            配置文件路径

        Returns
        -------
        dict
            实验结果
        """
        if not HAS_AEPSYCH_SERVER:
            print(f"错误: AEPsych Server模块未安装")
            return None

        print(f"\n{'='*80}")
        print(f" 运行 {method_name} 实验")
        print(f"{'='*80}\n")

        # 创建独立的虚拟用户实例
        user = VirtualUser(
            user_type=self.user_type, noise_level=self.user_noise, seed=self.seed
        )

        # 数据库路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = str(self.results_dir / f"experiment_{method_name}_{timestamp}.db")

        # 创建server
        server = AEPsychServer(database_path=db_path)

        # 加载配置
        full_config_path = Path(__file__).parent / config_path
        with open(full_config_path, "r", encoding="utf-8") as f:
            config_str = f.read()

        config_obj = Config(config_str=config_str)
        configure(server, config=config_obj)

        print(f"配置: {config_path}")
        print(f"数据库: {db_path}\n")

        # 实验数据
        trial_data = []
        trial_count = 0
        phase = "initialization"

        # 运行实验循环
        while not server.strat.finished:
            trial_count += 1

            # 获取下一个试验点
            config_dict = ask(server)
            if config_dict is None:
                break

            # 转换格式
            config_values = {
                key: val[0] if isinstance(val, list) else val
                for key, val in config_dict.items()
            }
            design = self._config_values_to_design(config_values)

            # 虚拟用户评分
            result = user.rate_design(design)
            rating = result["rating"]

            # 告诉server结果
            tell(server, config=config_dict, outcome=rating)

            # 检查阶段转换
            if trial_count == 15 and phase == "initialization":
                phase = "optimization"
                print(f"\n✓ 初始化阶段完成 (15次试验)")
                print(
                    f"  评分范围: {min([t['rating'] for t in user.trial_history])}-{max([t['rating'] for t in user.trial_history])}"
                )
                print(f"\n开始优化阶段...")

            # 记录数据
            trial_data.append(
                {
                    "trial": trial_count,
                    "phase": phase,
                    "design": design,
                    "rating": rating,
                    "true_score": user.get_ground_truth(design),
                    "rt": result["rt"],
                }
            )

            # 进度报告
            if trial_count % 10 == 0:
                print(f"  试验 {trial_count}/60 完成...")

        print(f"\n✓ {method_name} 实验完成!")
        print(f"  总试验数: {trial_count}")

        # 计算统计量
        result = {
            "method": method_name,
            "config_path": config_path,
            "db_path": db_path,
            "n_trials": trial_count,
            "trial_data": trial_data,
            "user": user,
        }

        return result

    def analyze_coverage(self, trial_data: List[Dict]) -> Dict:
        """
        分析采样覆盖

        Parameters
        ----------
        trial_data : list
            试验数据

        Returns
        -------
        dict
            覆盖统计
        """
        # 统计唯一设计
        unique_designs = set()
        for trial in trial_data:
            design = trial["design"]
            key = tuple(self._design_to_config_values(design))
            unique_designs.add(key)

        # 统计变量覆盖
        colors_sampled = set()
        layouts_sampled = set()
        fonts_sampled = set()
        animations_sampled = set()

        for trial in trial_data:
            design = trial["design"]
            colors_sampled.add(design["color_scheme"])
            layouts_sampled.add(design["layout"])
            fonts_sampled.add(design["font_size"])
            animations_sampled.add(design["animation"])

        # 计算交互项覆盖
        interaction_pairs = set()
        for trial in trial_data:
            design = trial["design"]
            # color x layout
            interaction_pairs.add((design["color_scheme"], design["layout"]))
            # layout x animation
            interaction_pairs.add((design["layout"], design["animation"]))
            # font x animation
            interaction_pairs.add((design["font_size"], design["animation"]))

        # 理论上的交互项数量
        max_color_layout = len(VirtualUser.COLOR_SCHEMES) * len(VirtualUser.LAYOUTS)
        max_layout_anim = len(VirtualUser.LAYOUTS) * len(VirtualUser.ANIMATIONS)
        max_font_anim = len(VirtualUser.FONT_SIZES) * len(VirtualUser.ANIMATIONS)

        coverage = {
            "unique_designs": len(unique_designs),
            "total_designs": VirtualUser.design_space_size(),
            "coverage_rate": len(unique_designs) / VirtualUser.design_space_size(),
            "repeat_rate": 1 - len(unique_designs) / len(trial_data),
            "color_coverage": len(colors_sampled) / len(VirtualUser.COLOR_SCHEMES),
            "layout_coverage": len(layouts_sampled) / len(VirtualUser.LAYOUTS),
            "font_coverage": len(fonts_sampled) / len(VirtualUser.FONT_SIZES),
            "animation_coverage": len(animations_sampled) / len(VirtualUser.ANIMATIONS),
            "avg_factor_coverage": np.mean(
                [
                    len(colors_sampled) / len(VirtualUser.COLOR_SCHEMES),
                    len(layouts_sampled) / len(VirtualUser.LAYOUTS),
                    len(fonts_sampled) / len(VirtualUser.FONT_SIZES),
                    len(animations_sampled) / len(VirtualUser.ANIMATIONS),
                ]
            ),
            "interaction_pairs": len(interaction_pairs),
            "color_layout_pairs": sum(
                1
                for pair in interaction_pairs
                if pair[0] in VirtualUser.COLOR_SCHEMES
                and pair[1] in VirtualUser.LAYOUTS
            ),
            "layout_anim_pairs": sum(
                1
                for pair in interaction_pairs
                if pair[0] in VirtualUser.LAYOUTS and pair[1] in VirtualUser.ANIMATIONS
            ),
            "font_anim_pairs": sum(
                1
                for pair in interaction_pairs
                if pair[0] in VirtualUser.FONT_SIZES
                and pair[1] in VirtualUser.ANIMATIONS
            ),
            "interaction_coverage": len(interaction_pairs)
            / (max_color_layout + max_layout_anim + max_font_anim),
        }

        return coverage

    def analyze_quality(self, trial_data: List[Dict]) -> Dict:
        """
        分析质量发现

        Parameters
        ----------
        trial_data : list
            试验数据

        Returns
        -------
        dict
            质量指标
        """
        ratings = np.array([t["rating"] for t in trial_data])
        true_scores = np.array([t["true_score"] for t in trial_data])

        # 高分发现
        high_score_threshold = 9.5
        n_high_scores = np.sum(true_scores >= high_score_threshold)

        quality = {
            "mean_rating": float(np.mean(ratings)),
            "std_rating": float(np.std(ratings)),
            "mean_true_score": float(np.mean(true_scores)),
            "std_true_score": float(np.std(true_scores)),
            "max_rating": int(np.max(ratings)),
            "max_true_score": float(np.max(true_scores)),
            "high_score_count": int(n_high_scores),
            "high_score_rate": float(n_high_scores / len(trial_data)),
        }

        return quality

    def run_all_experiments(self):
        """运行所有对比实验"""
        print(f"\n{'='*80}")
        print(f" 开始运行所有实验")
        print(f"{'='*80}\n")

        for method_name, config_file in self.configs.items():
            result = self.run_single_experiment(method_name, config_file)
            if result:
                # 分析结果
                coverage = self.analyze_coverage(result["trial_data"])
                quality = self.analyze_quality(result["trial_data"])

                result["coverage"] = coverage
                result["quality"] = quality

                self.all_results[method_name] = result

                print(f"\n{method_name} 关键指标:")
                print(f"  唯一设计: {coverage['unique_designs']}/80")
                print(f"  重复率: {coverage['repeat_rate']*100:.1f}%")
                print(f"  平均分数: {quality['mean_true_score']:.2f}")
                print(f"  高分发现: {quality['high_score_count']}")

    def generate_comparison_report(self):
        """生成对比报告"""
        print(f"\n{'='*80}")
        print(f" 生成对比报告")
        print(f"{'='*80}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建对比表格
        comparison_data = []
        for method_name in ["V1", "V3A", "V3C"]:
            if method_name not in self.all_results:
                continue

            result = self.all_results[method_name]
            coverage = result["coverage"]
            quality = result["quality"]

            comparison_data.append(
                {
                    "Method": method_name,
                    "Unique Designs": coverage["unique_designs"],
                    "Repeat Rate (%)": f"{coverage['repeat_rate']*100:.1f}",
                    "Coverage Rate (%)": f"{coverage['coverage_rate']*100:.1f}",
                    "Factor Coverage (%)": f"{coverage['avg_factor_coverage']*100:.1f}",
                    "Interaction Coverage (%)": f"{coverage['interaction_coverage']*100:.1f}",
                    "Mean Score": f"{quality['mean_true_score']:.2f}",
                    "Std Score": f"{quality['std_true_score']:.2f}",
                    "Max Score": f"{quality['max_true_score']:.2f}",
                    "High Score Count": quality["high_score_count"],
                }
            )

        df_comparison = pd.DataFrame(comparison_data)

        # 打印对比表
        print("\n对比结果:")
        print("=" * 120)
        print(df_comparison.to_string(index=False))
        print("=" * 120)

        # 保存CSV
        csv_path = self.results_dir / f"comparison_report_{timestamp}.csv"
        df_comparison.to_csv(csv_path, index=False)
        print(f"\n✓ 对比报告已保存: {csv_path}")

        # 保存JSON
        json_data = {
            "timestamp": timestamp,
            "user_type": self.user_type,
            "user_noise": self.user_noise,
            "seed": self.seed,
            "methods": {},
        }

        for method_name, result in self.all_results.items():
            json_data["methods"][method_name] = {
                "config": result["config_path"],
                "db_path": result["db_path"],
                "n_trials": result["n_trials"],
                "coverage": {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in result["coverage"].items()
                },
                "quality": {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in result["quality"].items()
                },
            }

        json_path = self.results_dir / f"comparison_report_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON报告已保存: {json_path}")

        return df_comparison, json_data

    def generate_visualizations(self):
        """生成可视化对比"""
        print(f"\n{'='*80}")
        print(f" 生成可视化")
        print(f"{'='*80}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建大图
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        methods = ["V1", "V3A", "V3C"]
        colors_map = {"V1": "#3498db", "V3A": "#2ecc71", "V3C": "#9b59b6"}

        # 1. 唯一设计数对比
        ax1 = fig.add_subplot(gs[0, 0])
        unique_counts = [
            self.all_results[m]["coverage"]["unique_designs"] for m in methods
        ]
        bars = ax1.bar(methods, unique_counts, color=[colors_map[m] for m in methods])
        ax1.set_ylabel("Unique Designs")
        ax1.set_title("Unique Design Count")
        ax1.set_ylim([0, 90])
        ax1.axhline(y=80, color="r", linestyle="--", label="Max (80)")
        for bar, val in zip(bars, unique_counts):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(val),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 重复率对比
        ax2 = fig.add_subplot(gs[0, 1])
        repeat_rates = [
            self.all_results[m]["coverage"]["repeat_rate"] * 100 for m in methods
        ]
        bars = ax2.bar(methods, repeat_rates, color=[colors_map[m] for m in methods])
        ax2.set_ylabel("Repeat Rate (%)")
        ax2.set_title("Repeat Sampling Rate")
        ax2.set_ylim([0, max(repeat_rates) * 1.2 if max(repeat_rates) > 0 else 10])
        for bar, val in zip(bars, repeat_rates):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )
        ax2.grid(True, alpha=0.3)

        # 3. 因子覆盖率对比
        ax3 = fig.add_subplot(gs[0, 2])
        factor_coverage = [
            self.all_results[m]["coverage"]["avg_factor_coverage"] * 100
            for m in methods
        ]
        bars = ax3.bar(methods, factor_coverage, color=[colors_map[m] for m in methods])
        ax3.set_ylabel("Factor Coverage (%)")
        ax3.set_title("Average Factor Level Coverage")
        ax3.set_ylim([0, 110])
        for bar, val in zip(bars, factor_coverage):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )
        ax3.grid(True, alpha=0.3)

        # 4. 交互项覆盖率对比
        ax4 = fig.add_subplot(gs[0, 3])
        interaction_coverage = [
            self.all_results[m]["coverage"]["interaction_coverage"] * 100
            for m in methods
        ]
        bars = ax4.bar(
            methods, interaction_coverage, color=[colors_map[m] for m in methods]
        )
        ax4.set_ylabel("Interaction Coverage (%)")
        ax4.set_title("Interaction Term Coverage")
        ax4.set_ylim([0, 110])
        for bar, val in zip(bars, interaction_coverage):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )
        ax4.grid(True, alpha=0.3)

        # 5. 平均分数对比
        ax5 = fig.add_subplot(gs[1, 0])
        mean_scores = [
            self.all_results[m]["quality"]["mean_true_score"] for m in methods
        ]
        bars = ax5.bar(methods, mean_scores, color=[colors_map[m] for m in methods])
        ax5.set_ylabel("Mean True Score")
        ax5.set_title("Average True Score")
        ax5.set_ylim([0, max(mean_scores) * 1.2])
        for bar, val in zip(bars, mean_scores):
            ax5.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )
        ax5.grid(True, alpha=0.3)

        # 6. 高分发现对比
        ax6 = fig.add_subplot(gs[1, 1])
        high_score_counts = [
            self.all_results[m]["quality"]["high_score_count"] for m in methods
        ]
        bars = ax6.bar(
            methods, high_score_counts, color=[colors_map[m] for m in methods]
        )
        ax6.set_ylabel("High Score Count")
        ax6.set_title("High Score Discovery (≥9.5)")
        ax6.set_ylim(
            [0, max(high_score_counts) * 1.3 if max(high_score_counts) > 0 else 10]
        )
        for bar, val in zip(bars, high_score_counts):
            ax6.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(val),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        ax6.grid(True, alpha=0.3)

        # 7. 分数分布对比
        ax7 = fig.add_subplot(gs[1, 2:])
        for i, method in enumerate(methods):
            trial_data = self.all_results[method]["trial_data"]
            ratings = [t["rating"] for t in trial_data]
            ax7.hist(
                ratings,
                bins=np.arange(0.5, 11.5, 1),
                alpha=0.5,
                label=method,
                color=colors_map[method],
            )
        ax7.set_xlabel("Rating")
        ax7.set_ylabel("Frequency")
        ax7.set_title("Rating Distribution Comparison")
        ax7.set_xticks(range(1, 11))
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8-10. 各方案的评分随试次变化
        for idx, method in enumerate(methods):
            ax = fig.add_subplot(gs[2, idx])
            trial_data = self.all_results[method]["trial_data"]
            trials = [t["trial"] for t in trial_data]
            ratings = [t["rating"] for t in trial_data]

            # 滚动平均
            window = 10
            if len(ratings) >= window:
                rolling_mean = pd.Series(ratings).rolling(window=window).mean()
                ax.plot(
                    trials,
                    rolling_mean,
                    color=colors_map[method],
                    linewidth=2,
                    label=f"MA({window})",
                )

            ax.scatter(trials, ratings, alpha=0.3, s=20, color=colors_map[method])
            ax.axvline(x=20, color="red", linestyle="--", alpha=0.5, label="Phase 2")
            ax.set_xlabel("Trial")
            ax.set_ylabel("Rating")
            ax.set_title(f"{method} - Rating Over Time")
            ax.set_ylim([0, 11])
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 11. 综合雷达图 (需要额外一列)
        ax11 = fig.add_subplot(gs[2, 3], projection="polar")

        # 标准化指标到0-1
        categories = [
            "Unique\nDesigns",
            "Factor\nCoverage",
            "Interaction\nCoverage",
            "Mean\nScore",
            "High Score\nDiscovery",
        ]
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        for method in methods:
            cov = self.all_results[method]["coverage"]
            qual = self.all_results[method]["quality"]

            values = [
                cov["unique_designs"] / 80,  # 归一化到0-1
                cov["avg_factor_coverage"],
                cov["interaction_coverage"],
                qual["mean_true_score"] / 12,  # 假设最高12分
                qual["high_score_count"] / 20,  # 假设最多20个高分
            ]
            values += values[:1]

            ax11.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=method,
                color=colors_map[method],
            )
            ax11.fill(angles, values, alpha=0.15, color=colors_map[method])

        ax11.set_xticks(angles[:-1])
        ax11.set_xticklabels(categories, size=9)
        ax11.set_ylim(0, 1)
        ax11.set_title("Comprehensive Performance", size=11, y=1.08)
        ax11.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax11.grid(True)

        plt.suptitle(
            "V3 Full Comparison Experiment Results",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # 保存
        output_path = self.results_dir / f"comparison_visualization_{timestamp}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ 可视化已保存: {output_path}")


def main():
    """主函数"""
    print(f"\n{'='*80}")
    print(f" V3完整对比实验启动")
    print(f"{'='*80}\n")

    # 创建实验
    experiment = V3ComparisonExperiment(user_type="balanced", user_noise=0.5, seed=42)

    # 运行所有实验
    experiment.run_all_experiments()

    # 生成报告
    experiment.generate_comparison_report()

    # 生成可视化
    experiment.generate_visualizations()

    print(f"\n{'='*80}")
    print(f" ✅ 所有实验和分析完成!")
    print(f"{'='*80}\n")
    print(f"结果保存在: {experiment.results_dir}")
    print(f"  - 数据库文件: experiment_*.db (3个)")
    print(f"  - 对比报告: comparison_report_*.csv")
    print(f"  - JSON数据: comparison_report_*.json")
    print(f"  - 可视化: comparison_visualization_*.png")


if __name__ == "__main__":
    main()

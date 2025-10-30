"""
分类/有序离散变量实验 - 使用AEPsych数据库保存

UI设计偏好评估实验
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


class CategoricalExperimentRunner:
    """
    分类/有序离散变量实验运行器

    使用AEPsych的数据库功能保存数据
    """

    def __init__(
        self,
        config_path: str,
        user_type: str = "balanced",
        user_noise: float = 0.5,
        db_path: Optional[str] = None,
        seed: int = 42,
    ):
        """
        初始化实验

        Parameters
        ----------
        config_path : str
            配置文件路径
        user_type : str
            虚拟用户类型
        user_noise : float
            用户评分噪声
        db_path : str, optional
            数据库文件路径,默认在results目录
        seed : int
            随机种子
        """
        self.config_path = config_path
        self.user_type = user_type
        self.user_noise = user_noise
        self.seed = seed

        # 创建results目录
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        # 数据库路径
        if db_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.db_path = str(self.results_dir / f"experiment_{timestamp}.db")
        else:
            self.db_path = db_path

        # 创建虚拟用户
        self.user = VirtualUser(user_type=user_type, noise_level=user_noise, seed=seed)

        # 实验数据
        self.experiment_id = None
        self.trial_data = []
        self.n_trials = 0

        print(f"\n{'='*80}")
        print(f" 分类/有序离散变量实验")
        print(f"{'='*80}")
        print(f"\n虚拟用户配置:")
        print(f"  - 用户类型: {user_type}")
        print(f"  - 噪声水平: {user_noise}")
        print(f"  - 设计空间: {VirtualUser.design_space_size()} 种组合")
        print(f"  - 采样预算: 80 次 ({80/VirtualUser.design_space_size()*100:.1f}%)")
        print(f"\n数据库: {self.db_path}")

    def _design_to_config_values(self, design: Dict) -> List:
        """
        将设计字典转换为配置值列表

        Parameters
        ----------
        design : dict
            设计参数字典

        Returns
        -------
        list
            [color_scheme, layout, font_size, animation]
        """
        return [
            design["color_scheme"],
            design["layout"],
            design["font_size"],
            design["animation"],
        ]

    def _config_values_to_design(self, values) -> Dict:
        """
        将配置值转换为设计字典

        Parameters
        ----------
        values : list or dict
            如果是list: [color_scheme, layout, font_size, animation]
            如果是dict: {'color_scheme': ..., 'layout': ..., ...}

        Returns
        -------
        dict
            设计参数字典
        """
        if isinstance(values, dict):
            # AEPsych Server返回的格式
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
            # 手动模式格式(list)
            return {
                "color_scheme": values[0],
                "layout": values[1],
                "font_size": (
                    int(values[2]) if not isinstance(values[2], str) else values[2]
                ),
                "animation": values[3],
            }

    def run_experiment_with_aepsych_server(self):
        """
        使用AEPsych Server运行实验(标准方式)
        """
        if not HAS_AEPSYCH_SERVER:
            print("错误: AEPsych数据库模块未安装")
            return

        print(f"\n{'='*80}")
        print(f" 使用AEPsych Server运行实验")
        print(f"{'='*80}\n")

        # 创建server
        server = AEPsychServer(database_path=self.db_path)

        # 加载配置(使用UTF-8编码)
        with open(self.config_path, "r", encoding="utf-8") as f:
            config_str = f.read()

        # 使用configure函数配置server
        config_obj = Config(config_str=config_str)
        configure(server, config=config_obj)

        print("实验开始...")
        print(f"配置: {self.config_path}")
        print(f"数据库: {self.db_path}\n")

        trial_count = 0
        phase = "initialization"

        # 运行实验循环
        while not server.strat.finished:
            trial_count += 1

            # 获取下一个试验点 - ask()返回字典{param_name: [value]}
            config_dict = ask(server)
            if config_dict is None:
                break

            # 将[value]格式转换为value格式用于虚拟用户
            config_values = {
                key: val[0] if isinstance(val, list) else val
                for key, val in config_dict.items()
            }

            # 转换为设计
            design = self._config_values_to_design(config_values)

            # 虚拟用户评分
            result = self.user.rate_design(design)
            rating = result["rating"]

            # 告诉server结果 - 使用原始config_dict(带列表的格式)
            tell(server, config=config_dict, outcome=rating)

            # 检查是否进入优化阶段
            if trial_count == 20 and phase == "initialization":
                phase = "optimization"
                print(f"\n✓ 初始化阶段完成 (20次试验)")
                print(
                    f"  评分范围: {min([t['rating'] for t in self.user.trial_history])}-{max([t['rating'] for t in self.user.trial_history])}"
                )
                print(f"\n{'='*80}")
                print(f" 优化阶段 (主动学习)")
                print(f"{'='*80}\n")

            # 记录数据
            self.trial_data.append(
                {
                    "trial": trial_count,
                    "phase": phase,
                    "design": design,
                    "rating": rating,
                    "true_score": self.user.get_ground_truth(design),
                    "rt": result["rt"],
                }
            )

            # 进度报告
            if trial_count % 10 == 0:
                print(f"  试验 {trial_count}/80 完成...")

        print(f"\n✓ 实验完成!")
        print(f"  总试验数: {trial_count}")
        print(f"  数据已保存到: {self.db_path}")

        self.n_trials = trial_count

        return server

    def run_experiment_manual(self):
        """
        手动运行实验(不使用server,用于兼容性)
        """
        print(f"\n{'='*80}")
        print(f" 手动模式运行实验")
        print(f"{'='*80}\n")

        # 获取所有可能的设计
        all_designs = self.user.get_all_designs()
        np.random.shuffle(all_designs)

        print("阶段1: 初始化 (20次随机采样)")

        # 初始化阶段
        for i in range(20):
            design = all_designs[i]
            result = self.user.rate_design(design)

            self.trial_data.append(
                {
                    "trial": i + 1,
                    "phase": "initialization",
                    "design": design,
                    "rating": result["rating"],
                    "true_score": self.user.get_ground_truth(design),
                    "rt": result["rt"],
                }
            )

            if (i + 1) % 5 == 0:
                print(f"  完成 {i+1}/20...")

        print(f"\n✓ 初始化完成")
        print(f"\n阶段2: 优化 (60次采样)")

        # 优化阶段(继续随机,实际应使用采集函数)
        for i in range(20, 80):
            design = all_designs[i]
            result = self.user.rate_design(design)

            self.trial_data.append(
                {
                    "trial": i + 1,
                    "phase": "optimization",
                    "design": design,
                    "rating": result["rating"],
                    "true_score": self.user.get_ground_truth(design),
                    "rt": result["rt"],
                }
            )

            if (i + 1) % 10 == 0:
                print(f"  完成 {i+1}/80...")

        print(f"\n✓ 实验完成!")
        self.n_trials = 80

    def evaluate_coverage(self) -> Dict:
        """
        评估采样覆盖情况

        Returns
        -------
        dict
            覆盖统计
        """
        print(f"\n{'='*80}")
        print(f" 评估采样覆盖")
        print(f"{'='*80}\n")

        # 统计每个变量的覆盖
        colors_sampled = set()
        layouts_sampled = set()
        fonts_sampled = set()
        animations_sampled = set()

        for trial in self.trial_data:
            design = trial["design"]
            colors_sampled.add(design["color_scheme"])
            layouts_sampled.add(design["layout"])
            fonts_sampled.add(design["font_size"])
            animations_sampled.add(design["animation"])

        coverage = {
            "color_scheme": {
                "sampled": len(colors_sampled),
                "total": len(VirtualUser.COLOR_SCHEMES),
                "coverage": len(colors_sampled) / len(VirtualUser.COLOR_SCHEMES),
            },
            "layout": {
                "sampled": len(layouts_sampled),
                "total": len(VirtualUser.LAYOUTS),
                "coverage": len(layouts_sampled) / len(VirtualUser.LAYOUTS),
            },
            "font_size": {
                "sampled": len(fonts_sampled),
                "total": len(VirtualUser.FONT_SIZES),
                "coverage": len(fonts_sampled) / len(VirtualUser.FONT_SIZES),
            },
            "animation": {
                "sampled": len(animations_sampled),
                "total": len(VirtualUser.ANIMATIONS),
                "coverage": len(animations_sampled) / len(VirtualUser.ANIMATIONS),
            },
        }

        print("各变量水平覆盖:")
        for var, stats in coverage.items():
            print(
                f"  {var:15} {stats['sampled']}/{stats['total']} ({stats['coverage']*100:.1f}%)"
            )

        # 唯一组合数
        unique_designs = len(
            set(
                [
                    (
                        t["design"]["color_scheme"],
                        t["design"]["layout"],
                        t["design"]["font_size"],
                        t["design"]["animation"],
                    )
                    for t in self.trial_data
                ]
            )
        )

        print(
            f"\n唯一设计组合: {unique_designs}/{VirtualUser.design_space_size()} ({unique_designs/VirtualUser.design_space_size()*100:.1f}%)"
        )

        coverage["unique_designs"] = unique_designs
        coverage["total_space"] = VirtualUser.design_space_size()

        return coverage

    def evaluate_prediction_accuracy(self) -> Dict:
        """
        评估预测精度

        Returns
        -------
        dict
            精度指标
        """
        print(f"\n{'='*80}")
        print(f" 评估预测精度")
        print(f"{'='*80}\n")

        # 在所有360个设计上评估真实偏好
        all_designs = self.user.get_all_designs()
        true_scores = []
        observed_ratings = {
            tuple(self._design_to_config_values(t["design"])): t["rating"]
            for t in self.trial_data
        }

        for design in all_designs:
            true_score = self.user.get_ground_truth(design)
            true_scores.append(true_score)

        true_scores = np.array(true_scores)

        # 对采样点,计算预测误差
        sampled_true = []
        sampled_observed = []

        for trial in self.trial_data:
            sampled_true.append(trial["true_score"])
            sampled_observed.append(trial["rating"])

        sampled_true = np.array(sampled_true)
        sampled_observed = np.array(sampled_observed)

        # 计算相关性(需要将评分映射回分数空间)
        # 粗略映射: rating(1-10) -> score(4-12)
        sampled_observed_mapped = (sampled_observed - 1) / 9 * 8 + 4

        correlation = np.corrcoef(sampled_true, sampled_observed_mapped)[0, 1]
        mae = np.mean(np.abs(sampled_true - sampled_observed_mapped))
        rmse = np.sqrt(np.mean((sampled_true - sampled_observed_mapped) ** 2))

        # 评分一致性(直接比较离散评分)
        # 将true_score转换为预期评分
        expected_ratings = np.clip(
            np.round((sampled_true - 4) / 8 * 9 + 1), 1, 10
        ).astype(int)
        exact_match = np.mean(expected_ratings == sampled_observed)
        within_1 = np.mean(np.abs(expected_ratings - sampled_observed) <= 1)
        within_2 = np.mean(np.abs(expected_ratings - sampled_observed) <= 2)

        metrics = {
            "correlation": correlation,
            "mae": mae,
            "rmse": rmse,
            "exact_match": exact_match,
            "within_1": within_1,
            "within_2": within_2,
            "n_samples": len(sampled_true),
        }

        print(f"采样点预测精度 (n={len(sampled_true)}):")
        print(f"  相关系数: {correlation:.4f}")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"\n评分一致性:")
        print(f"  完全匹配: {exact_match*100:.1f}%")
        print(f"  ±1以内:   {within_1*100:.1f}%")
        print(f"  ±2以内:   {within_2*100:.1f}%")

        # 找出最佳设计
        best_idx = np.argmax(true_scores)
        best_design = all_designs[best_idx]
        best_score = true_scores[best_idx]

        print(f"\n真实最佳设计:")
        print(f"  {best_design}")
        print(f"  真实分数: {best_score:.2f}")

        # 检查是否采样到
        best_key = tuple(self._design_to_config_values(best_design))
        if best_key in observed_ratings:
            print(f"  ✓ 已采样,评分: {observed_ratings[best_key]}/10")
        else:
            print(f"  ✗ 未采样")

        # 找出采样中评分最高的
        best_sampled_idx = np.argmax(sampled_observed)
        best_sampled_design = self.trial_data[best_sampled_idx]["design"]
        best_sampled_rating = sampled_observed[best_sampled_idx]
        best_sampled_true = sampled_true[best_sampled_idx]

        print(f"\n采样中评分最高:")
        print(f"  {best_sampled_design}")
        print(f"  用户评分: {best_sampled_rating}/10")
        print(f"  真实分数: {best_sampled_true:.2f}")

        metrics["best_design"] = best_design
        metrics["best_score"] = best_score
        metrics["best_sampled_design"] = best_sampled_design
        metrics["best_sampled_rating"] = int(best_sampled_rating)

        return metrics

    def save_additional_data(self, coverage: Dict, metrics: Dict):
        """
        保存额外的分析数据(CSV和JSON)

        Parameters
        ----------
        coverage : dict
            覆盖统计
        metrics : dict
            精度指标
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存试次数据为CSV
        rows = []
        for trial in self.trial_data:
            row = {
                "trial": trial["trial"],
                "phase": trial["phase"],
                "color_scheme": trial["design"]["color_scheme"],
                "layout": trial["design"]["layout"],
                "font_size": trial["design"]["font_size"],
                "animation": trial["design"]["animation"],
                "rating": trial["rating"],
                "true_score": trial["true_score"],
                "rt": trial["rt"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = self.results_dir / f"trial_data_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ 试次数据已保存: {csv_path}")

        # 保存元数据
        metadata = {
            "timestamp": timestamp,
            "user_type": self.user_type,
            "user_noise": self.user_noise,
            "seed": self.seed,
            "n_trials": self.n_trials,
            "design_space_size": VirtualUser.design_space_size(),
            "sampling_ratio": self.n_trials / VirtualUser.design_space_size(),
            "coverage": coverage,
            "metrics": {
                "correlation": float(metrics["correlation"]),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "exact_match": float(metrics["exact_match"]),
                "within_1": float(metrics["within_1"]),
                "within_2": float(metrics["within_2"]),
            },
            "best_design": metrics["best_design"],
            "best_score": float(metrics["best_score"]),
            "best_sampled_design": metrics["best_sampled_design"],
            "best_sampled_rating": int(metrics["best_sampled_rating"]),
        }

        json_path = self.results_dir / f"metadata_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ 元数据已保存: {json_path}")

        return csv_path, json_path

    def visualize_results(self, coverage: Dict, metrics: Dict):
        """
        可视化实验结果

        Parameters
        ----------
        coverage : dict
            覆盖统计
        metrics : dict
            精度指标
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 变量覆盖条形图
        ax1 = fig.add_subplot(gs[0, 0])
        vars_labels = ["color", "layout", "font", "animation"]
        coverages = [
            coverage["color_scheme"]["coverage"],
            coverage["layout"]["coverage"],
            coverage["font_size"]["coverage"],
            coverage["animation"]["coverage"],
        ]
        ax1.bar(
            vars_labels, coverages, color=["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
        )
        ax1.set_ylim([0, 1.1])
        ax1.set_ylabel("Coverage Ratio")
        ax1.set_title("Variable Level Coverage")
        ax1.grid(True, alpha=0.3)

        # 2. 评分分布
        ax2 = fig.add_subplot(gs[0, 1])
        ratings = [t["rating"] for t in self.trial_data]
        ax2.hist(
            ratings, bins=np.arange(0.5, 11.5, 1), edgecolor="black", color="skyblue"
        )
        ax2.set_xlabel("Rating")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"Rating Distribution (n={len(ratings)})")
        ax2.set_xticks(range(1, 11))
        ax2.grid(True, alpha=0.3)

        # 3. 真实分数 vs 观测评分
        ax3 = fig.add_subplot(gs[0, 2])
        true_scores = [t["true_score"] for t in self.trial_data]
        observed_mapped = [(r - 1) / 9 * 8 + 4 for r in ratings]
        ax3.scatter(true_scores, observed_mapped, alpha=0.5, s=30)
        ax3.plot(
            [min(true_scores), max(true_scores)],
            [min(true_scores), max(true_scores)],
            "r--",
            linewidth=2,
            label="Perfect",
        )
        ax3.set_xlabel("True Score")
        ax3.set_ylabel("Observed Score (mapped)")
        ax3.set_title(f'True vs Observed (r={metrics["correlation"]:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 阶段对比
        ax4 = fig.add_subplot(gs[1, 0])
        init_ratings = [
            t["rating"] for t in self.trial_data if t["phase"] == "initialization"
        ]
        opt_ratings = [
            t["rating"] for t in self.trial_data if t["phase"] == "optimization"
        ]

        ax4.boxplot([init_ratings, opt_ratings], labels=["Init (20)", "Opt (60)"])
        ax4.set_ylabel("Rating")
        ax4.set_title("Rating Distribution by Phase")
        ax4.grid(True, alpha=0.3)

        # 5. 色彩方案采样
        ax5 = fig.add_subplot(gs[1, 1])
        color_counts = {}
        for t in self.trial_data:
            c = t["design"]["color_scheme"]
            color_counts[c] = color_counts.get(c, 0) + 1

        colors = list(color_counts.keys())
        counts = list(color_counts.values())
        ax5.bar(
            colors,
            counts,
            color=["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"],
        )
        ax5.set_ylabel("Frequency")
        ax5.set_title("Color Scheme Sampling")
        ax5.grid(True, alpha=0.3)

        # 6. 布局采样
        ax6 = fig.add_subplot(gs[1, 2])
        layout_counts = {}
        for t in self.trial_data:
            l = t["design"]["layout"]
            layout_counts[l] = layout_counts.get(l, 0) + 1

        layouts = list(layout_counts.keys())
        counts = list(layout_counts.values())
        ax6.bar(layouts, counts, color="teal")
        ax6.set_ylabel("Frequency")
        ax6.set_title("Layout Sampling")
        ax6.grid(True, alpha=0.3)

        # 7. 字体大小采样
        ax7 = fig.add_subplot(gs[2, 0])
        font_counts = {}
        for t in self.trial_data:
            f = t["design"]["font_size"]
            font_counts[f] = font_counts.get(f, 0) + 1

        fonts = sorted(font_counts.keys())
        counts = [font_counts[f] for f in fonts]
        ax7.bar([str(f) for f in fonts], counts, color="coral")
        ax7.set_ylabel("Frequency")
        ax7.set_xlabel("Font Size")
        ax7.set_title("Font Size Sampling")
        ax7.grid(True, alpha=0.3)

        # 8. 动画采样
        ax8 = fig.add_subplot(gs[2, 1])
        anim_counts = {}
        for t in self.trial_data:
            a = t["design"]["animation"]
            anim_counts[a] = anim_counts.get(a, 0) + 1

        anims = list(anim_counts.keys())
        counts = list(anim_counts.values())
        ax8.bar(anims, counts, color="mediumpurple")
        ax8.set_ylabel("Frequency")
        ax8.set_title("Animation Sampling")
        ax8.grid(True, alpha=0.3)

        # 9. 预测精度指标
        ax9 = fig.add_subplot(gs[2, 2])
        metric_names = ["Exact\nMatch", "±1", "±2"]
        metric_values = [
            metrics["exact_match"],
            metrics["within_1"],
            metrics["within_2"],
        ]
        colors_bars = ["#e74c3c", "#f39c12", "#2ecc71"]
        bars = ax9.bar(metric_names, metric_values, color=colors_bars)
        ax9.set_ylim([0, 1.1])
        ax9.set_ylabel("Proportion")
        ax9.set_title("Prediction Accuracy")

        # 添加数值标签
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax9.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val*100:.1f}%",
                ha="center",
                va="bottom",
            )
        ax9.grid(True, alpha=0.3)

        plt.suptitle(
            "Categorical Experiment Results", fontsize=16, fontweight="bold", y=0.995
        )

        output_path = self.results_dir / f"results_visualization_{timestamp}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ 可视化已保存: {output_path}")


def main():
    """主函数"""
    # 配置路径
    config_path = Path(__file__).parent / "experiment_config.ini"

    # 创建实验
    exp = CategoricalExperimentRunner(
        config_path=str(config_path), user_type="balanced", user_noise=0.5, seed=42
    )

    # 尝试使用AEPsych Server
    if HAS_AEPSYCH_SERVER:
        try:
            server = exp.run_experiment_with_aepsych_server()
        except Exception as e:
            import traceback

            print(f"\n警告: AEPsych Server运行失败: {e}")
            print("\n详细错误:\n")
            traceback.print_exc()
            print("回退到手动模式...\n")
            exp.run_experiment_manual()
    else:
        print("\n使用手动模式运行(AEPsych DB未完全可用)...\n")
        exp.run_experiment_manual()

    # 评估
    if exp.n_trials > 0:
        coverage = exp.evaluate_coverage()
        metrics = exp.evaluate_prediction_accuracy()

        # 保存数据
        exp.save_additional_data(coverage, metrics)

        # 可视化
        exp.visualize_results(coverage, metrics)

        print(f"\n{'='*80}")
        print(f" 实验完成!")
        print(f"{'='*80}")
        print(f"\n所有结果已保存到: {exp.results_dir}")
    else:
        print("\n错误: 没有运行任何试验")


if __name__ == "__main__":
    main()

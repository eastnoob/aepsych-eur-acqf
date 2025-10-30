"""
分类/有序离散变量实验 V2 - 使用改进的采集函数

改进要点:
1. 使用EnhancedVarianceReductionAcqf避免重复采样
2. 强化空间覆盖和多样性
3. 分区均匀性保证
4. 动态权重调整
5. 高分区域适度exploration
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

    # 重要: 确保新的采集函数可以被导入
    from dynamic_eur_acquisition.acquisition_function_v2 import (
        EnhancedVarianceReductionAcqf,
    )

    Config.register_module(dynamic_eur_acquisition)

    HAS_AEPSYCH_SERVER = True
except ImportError as e:
    print(f"警告: 无法导入AEPsych Server模块: {e}")
    HAS_AEPSYCH_SERVER = False


class CategoricalExperimentRunnerV2:
    """
    分类/有序离散变量实验运行器 V2

    使用改进的采集函数
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
            数据库文件路径,默认在results_v2目录
        seed : int
            随机种子
        """
        self.config_path = config_path
        self.user_type = user_type
        self.user_noise = user_noise
        self.seed = seed

        # 创建results_v2目录
        self.results_dir = Path(__file__).parent / "results_v2"
        self.results_dir.mkdir(exist_ok=True)

        # 数据库路径
        if db_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.db_path = str(self.results_dir / f"experiment_v2_{timestamp}.db")
        else:
            self.db_path = db_path

        # 创建虚拟用户
        self.user = VirtualUser(user_type=user_type, noise_level=user_noise, seed=seed)

        # 实验数据
        self.experiment_id = None
        self.trial_data = []
        self.n_trials = 0

        # 采集函数诊断数据
        self.acqf_diagnostics = []

        print(f"\n{'='*80}")
        print("分类/有序离散变量实验 V2 - 改进版采集函数")
        print(f"{'='*80}")
        print(f"配置文件: {config_path}")
        print(f"虚拟用户: {user_type} (噪声={user_noise})")
        print(f"数据库: {self.db_path}")
        print(f"随机种子: {seed}")
        print(f"{'='*80}\n")

    def setup_server(self):
        """设置AEPsych server"""
        # 读取配置文件
        with open(self.config_path, "r", encoding="utf-8") as f:
            config_str = f.read()

        # 创建server并配置
        self.server = AEPsychServer(database_path=self.db_path)

        # 使用Config对象配置server（与原始脚本一致）
        config_obj = Config(config_str=config_str)
        configure(self.server, config=config_obj)

        print("✓ Server配置成功")
        self.experiment_id = (
            self.server.get_experiment_id()
            if hasattr(self.server, "get_experiment_id")
            else None
        )
        if self.experiment_id:
            print(f"  实验ID: {self.experiment_id}")

        return {"success": True}

    def run_trial(self, trial_num: int) -> Dict:
        """
        运行单次trial

        Parameters
        ----------
        trial_num : int
            Trial编号(从1开始)

        Returns
        -------
        Dict
            Trial结果
        """
        # 检查是否完成
        if self.server.strat.finished:
            print(f"\n实验已完成 (共{self.n_trials}次trials)")
            return {"is_finished": True}

        # 1. 请求下一个采样点 - ask()返回字典{param_name: [value]}
        config_dict = ask(self.server)
        if config_dict is None:
            return {"is_finished": True}

        # 2. 将[value]格式转换为value格式用于虚拟用户
        config_values = {
            key: val[0] if isinstance(val, list) else val
            for key, val in config_dict.items()
        }

        # 3. 转换为设计字典
        design = {
            "color_scheme": config_values["color_scheme"],
            "layout": config_values["layout"],
            "font_size": (
                int(config_values["font_size"])
                if not isinstance(config_values["font_size"], str)
                else config_values["font_size"]
            ),
            "animation": config_values["animation"],
        }

        # 4. 获取用户评分
        result = self.user.rate_design(design)
        rating = result["rating"]
        true_score = self.user.get_ground_truth(design)

        # 5. 反馈结果给server - 使用原始config_dict(带列表的格式)
        tell(self.server, config=config_dict, outcome=rating)

        # 6. 记录trial数据
        trial_result = {
            "trial": trial_num,
            "color_scheme": design["color_scheme"],
            "layout": design["layout"],
            "font_size": design["font_size"],
            "animation": design["animation"],
            "rating": rating,
            "true_score": true_score,
            "rt": result["rt"],
            "is_finished": False,
        }

        self.trial_data.append(trial_result)
        self.n_trials += 1

        # 7. 尝试获取采集函数诊断信息
        try:
            if hasattr(self.server, "strat") and hasattr(
                self.server.strat, "generator"
            ):
                generator = self.server.strat.generator
                if hasattr(generator, "acqf"):
                    acqf = generator.acqf
                    if hasattr(acqf, "get_diagnostics"):
                        diagnostics = acqf.get_diagnostics()
                        diagnostics["trial"] = trial_num
                        self.acqf_diagnostics.append(diagnostics)
        except Exception as e:
            pass  # 静默忽略

        # 8. 打印进度
        print(
            f"Trial {trial_num:3d}: {design['color_scheme']:8s} | {design['layout']:8s} | "
            f"Font={design['font_size']:2d} | {design['animation']:8s} | "
            f"Rating={rating}/10 | True={true_score:.2f}"
        )

        return trial_result

    def run_experiment(self, max_trials: int = 80):
        """
        运行完整实验

        Parameters
        ----------
        max_trials : int
            最大trial数
        """
        print(f"\n开始实验 (最多{max_trials}次trials)...\n")

        for trial_num in range(1, max_trials + 1):
            result = self.run_trial(trial_num)

            if result.get("is_finished"):
                break

        print(f"\n{'='*80}")
        print(f"实验完成! 共进行了 {self.n_trials} 次trials")
        print(f"{'='*80}\n")

    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存trial数据为CSV
        df = pd.DataFrame(self.trial_data)
        csv_path = self.results_dir / f"trial_data_v2_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Trial数据已保存: {csv_path}")

        # 2. 保存元数据
        metadata = {
            "timestamp": timestamp,
            "config_path": str(self.config_path),
            "db_path": self.db_path,
            "experiment_id": self.experiment_id,
            "n_trials": self.n_trials,
            "user_type": self.user_type,
            "user_noise": self.user_noise,
            "seed": self.seed,
            "version": "v2",
        }

        # 添加统计信息
        if len(df) > 0:
            metadata["statistics"] = {
                "mean_rating": float(df["rating"].mean()),
                "std_rating": float(df["rating"].std()),
                "mean_true_score": float(df["true_score"].mean()),
                "std_true_score": float(df["true_score"].std()),
                "correlation": float(df["rating"].corr(df["true_score"])),
                "mae": float(np.mean(np.abs(df["rating"] - df["true_score"]))),
                "rmse": float(np.sqrt(np.mean((df["rating"] - df["true_score"]) ** 2))),
            }

            # 计算唯一设计数
            unique_designs = df[
                ["color_scheme", "layout", "font_size", "animation"]
            ].drop_duplicates()
            metadata["unique_designs"] = len(unique_designs)
            metadata["coverage_rate"] = len(unique_designs) / len(df)

        json_path = self.results_dir / f"metadata_v2_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ 元数据已保存: {json_path}")

        # 3. 保存采集函数诊断数据
        if self.acqf_diagnostics:
            diag_df = pd.DataFrame(self.acqf_diagnostics)
            diag_path = self.results_dir / f"acqf_diagnostics_v2_{timestamp}.csv"
            diag_df.to_csv(diag_path, index=False)
            print(f"✓ 采集函数诊断数据已保存: {diag_path}")

        return csv_path, json_path

    def create_visualization(self):
        """创建结果可视化"""
        if len(self.trial_data) == 0:
            print("警告: 没有数据可供可视化")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(self.trial_data)

        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"分类/有序离散变量实验结果 V2 (n={self.n_trials})",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Rating vs True Score
        ax = axes[0, 0]
        ax.scatter(df["true_score"], df["rating"], alpha=0.5, s=50)
        ax.plot(
            [df["true_score"].min(), df["true_score"].max()],
            [df["true_score"].min(), df["true_score"].max()],
            "r--",
            lw=2,
            label="Perfect prediction",
        )
        corr = df["rating"].corr(df["true_score"])
        ax.set_xlabel("True Score")
        ax.set_ylabel("Rating")
        ax.set_title(f"Rating vs True Score (r={corr:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Rating and True Score over trials
        ax = axes[0, 1]
        ax.plot(df["trial"], df["rating"], "o-", label="Rating", alpha=0.6)
        ax.plot(df["trial"], df["true_score"], "s-", label="True Score", alpha=0.6)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        ax.set_title("Scores Over Trials")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Error over trials
        ax = axes[0, 2]
        error = df["rating"] - df["true_score"]
        ax.plot(df["trial"], error, "o-", alpha=0.6)
        ax.axhline(y=0, color="r", linestyle="--", lw=2)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Error (Rating - True Score)")
        ax.set_title(f"Prediction Error (MAE={np.abs(error).mean():.3f})")
        ax.grid(True, alpha=0.3)

        # 4. Color scheme distribution
        ax = axes[1, 0]
        color_counts = df["color_scheme"].value_counts()
        color_counts.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
        ax.set_xlabel("Color Scheme")
        ax.set_ylabel("Count")
        ax.set_title("Color Scheme Distribution")
        ax.tick_params(axis="x", rotation=45)

        # 5. Layout distribution
        ax = axes[1, 1]
        layout_counts = df["layout"].value_counts()
        layout_counts.plot(kind="bar", ax=ax, color="lightcoral", edgecolor="black")
        ax.set_xlabel("Layout")
        ax.set_ylabel("Count")
        ax.set_title("Layout Distribution")
        ax.tick_params(axis="x", rotation=45)

        # 6. Font size distribution
        ax = axes[1, 2]
        font_counts = df["font_size"].value_counts().sort_index()
        font_counts.plot(kind="bar", ax=ax, color="lightgreen", edgecolor="black")
        ax.set_xlabel("Font Size")
        ax.set_ylabel("Count")
        ax.set_title("Font Size Distribution")
        ax.tick_params(axis="x", rotation=0)

        plt.tight_layout()

        # 保存图表
        viz_path = self.results_dir / f"results_visualization_v2_{timestamp}.png"
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ 可视化已保存: {viz_path}")
        return viz_path


def main():
    """主函数"""
    if not HAS_AEPSYCH_SERVER:
        print("错误: 需要安装AEPsych Server")
        return

    # 配置路径
    config_path = Path(__file__).parent / "experiment_config_v2.ini"

    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return

    # 创建并运行实验
    runner = CategoricalExperimentRunnerV2(
        config_path=str(config_path),
        user_type="balanced",  # 可选: balanced, high_preference, low_preference
        user_noise=0.5,  # 评分噪声水平
        seed=42,
    )

    try:
        # 设置server
        runner.setup_server()

        # 运行实验
        runner.run_experiment(max_trials=80)

        # 保存结果
        runner.save_results()

        # 创建可视化
        runner.create_visualization()

        print("\n✓ 实验完成!")

    except Exception as e:
        print(f"\n✗ 实验出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

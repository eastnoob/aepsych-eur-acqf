# -*- coding: utf-8 -*-
"""
完整模拟实验 - 使用标准AEPsych流程

这个脚本模拟一个完整的心理物理学实验:
1. 虚拟被试根据真实的心理物理函数响应
2. 使用标准AEPsych流程(初始化+优化)
3. 使用我们的VarianceReductionWithCoverageAcqf采集函数
4. 保存完整的实验数据和分析结果

运行: pixi run python run_experiment.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from acquisition_function import VarianceReductionWithCoverageAcqf
from virtual_subject import VirtualSubject


class ExperimentRunner:
    """
    实验运行器 - 管理整个实验流程
    """

    def __init__(
        self,
        config_path: str = "experiment_config.ini",
        subject_type: str = "nonlinear_interaction",
        subject_noise: float = 0.3,
        save_dir: str = "results",
        seed: int = 42,
    ):
        """
        初始化实验

        Parameters
        ----------
        config_path : str
            实验配置文件路径
        subject_type : str
            被试类型(真实函数类型)
        subject_noise : float
            被试噪声水平
        save_dir : str
            结果保存目录
        seed : int
            随机种子
        """
        self.config_path = Path(__file__).parent / config_path
        self.save_dir = Path(__file__).parent / save_dir
        self.save_dir.mkdir(exist_ok=True)
        self.seed = seed

        np.random.seed(seed)

        # 创建虚拟被试
        self.subject = VirtualSubject(
            true_function_type=subject_type,
            noise_std=subject_noise,
            response_type="continuous",
            seed=seed,
        )

        # 创建采集函数
        self.acq_fn = VarianceReductionWithCoverageAcqf(
            config_ini_path=self.config_path if self.config_path.exists() else None,
            interaction_terms=[(0, 1), (1, 2)],  # intensity×duration, duration×noise
        )

        # 实验数据
        self.X_train = None
        self.y_train = None
        self.history = {
            "trial": [],
            "phase": [],
            "x1": [],
            "x2": [],
            "x3": [],
            "response": [],
            "true_value": [],
            "lambda_t": [],
            "r_t": [],
            "acq_score": [],
            "timestamp": [],
        }

        # 参数边界
        self.bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        self.param_names = ["intensity", "duration", "noise_level"]

        print("=" * 80)
        print(" 实验初始化完成 ")
        print("=" * 80)
        print(f"配置文件: {self.config_path}")
        print(f"被试类型: {subject_type}")
        print(f"噪声水平: {subject_noise}")
        print(f"结果保存: {self.save_dir}")
        print(f"采集函数参数:")
        print(f"  - lambda: [{self.acq_fn.lambda_min}, {self.acq_fn.lambda_max}]")
        print(f"  - tau: [{self.acq_fn.tau_2}, {self.acq_fn.tau_1}]")
        print(f"  - gamma: {self.acq_fn.gamma}")
        print(f"  - 交互项: {self.acq_fn.interaction_terms}")

    def _sobol_sample(self, n_samples: int) -> np.ndarray:
        """生成Sobol准随机样本"""
        try:
            from scipy.stats import qmc

            sampler = qmc.Sobol(d=3, scramble=True, seed=self.seed)
            samples = sampler.random(n_samples)
        except ImportError:
            samples = np.random.rand(n_samples, 3)

        # 缩放到边界
        for i in range(3):
            samples[:, i] = self.bounds[i, 0] + samples[:, i] * (
                self.bounds[i, 1] - self.bounds[i, 0]
            )

        return samples

    def run_initialization_phase(self, n_init: int = 20):
        """
        阶段1: 初始化 - 使用Sobol采样

        Parameters
        ----------
        n_init : int
            初始采样数量
        """
        print("\n" + "=" * 80)
        print(f" 阶段1: 初始化 (n={n_init}) ")
        print("=" * 80)

        # 生成Sobol样本
        X_init = self._sobol_sample(n_init)

        # 从被试收集响应
        y_init = []
        for i, x in enumerate(X_init):
            # 准备刺激
            stimulus = {"x1": x[0], "x2": x[1], "x3": x[2]}

            # 被试响应
            response_data = self.subject.respond(stimulus)
            y_init.append(response_data["response"])

            # 记录历史
            self.history["trial"].append(i)
            self.history["phase"].append("initialization")
            self.history["x1"].append(x[0])
            self.history["x2"].append(x[1])
            self.history["x3"].append(x[2])
            self.history["response"].append(response_data["response"])
            self.history["true_value"].append(response_data["true_value"])
            self.history["lambda_t"].append(np.nan)
            self.history["r_t"].append(np.nan)
            self.history["acq_score"].append(np.nan)
            self.history["timestamp"].append(datetime.now().isoformat())

            if (i + 1) % 5 == 0:
                print(f"  完成 {i+1}/{n_init} 次试验...")

        self.X_train = X_init
        self.y_train = np.array(y_init)

        # 拟合初始模型
        self.acq_fn.fit(self.X_train, self.y_train)

        print(f"\n✓ 初始化完成!")
        print(f"  - 收集了 {n_init} 个初始数据点")
        print(f"  - 响应范围: [{self.y_train.min():.3f}, {self.y_train.max():.3f}]")
        print(f"  - 初始 λ_t = {self.acq_fn.get_current_lambda():.3f}")
        print(f"  - 初始 r_t = {self.acq_fn.get_variance_reduction_ratio():.3f}")

    def run_optimization_phase(self, n_opt: int = 40, n_candidates: int = 300):
        """
        阶段2: 优化 - 使用采集函数主动选点

        Parameters
        ----------
        n_opt : int
            优化迭代次数
        n_candidates : int
            每次迭代的候选点数
        """
        print("\n" + "=" * 80)
        print(f" 阶段2: 优化 (n={n_opt}) ")
        print("=" * 80)

        start_trial = len(self.history["trial"])

        for iteration in range(n_opt):
            # 生成候选点
            X_candidates = np.random.rand(n_candidates, 3)
            for i in range(3):
                X_candidates[:, i] = self.bounds[i, 0] + X_candidates[:, i] * (
                    self.bounds[i, 1] - self.bounds[i, 0]
                )

            # 评估采集函数
            acq_scores = self.acq_fn(X_candidates)

            # 选择最佳点
            best_idx = np.argmax(acq_scores)
            x_next = X_candidates[best_idx]
            best_score = acq_scores[best_idx]

            # 被试响应
            stimulus = {"x1": x_next[0], "x2": x_next[1], "x3": x_next[2]}
            response_data = self.subject.respond(stimulus)

            # 更新数据集
            self.X_train = np.vstack([self.X_train, x_next])
            self.y_train = np.append(self.y_train, response_data["response"])

            # 重新拟合模型
            self.acq_fn.fit(self.X_train, self.y_train)

            # 获取当前状态
            lambda_t = self.acq_fn.get_current_lambda()
            r_t = self.acq_fn.get_variance_reduction_ratio()

            # 记录历史
            trial_num = start_trial + iteration
            self.history["trial"].append(trial_num)
            self.history["phase"].append("optimization")
            self.history["x1"].append(x_next[0])
            self.history["x2"].append(x_next[1])
            self.history["x3"].append(x_next[2])
            self.history["response"].append(response_data["response"])
            self.history["true_value"].append(response_data["true_value"])
            self.history["lambda_t"].append(lambda_t)
            self.history["r_t"].append(r_t)
            self.history["acq_score"].append(best_score)
            self.history["timestamp"].append(datetime.now().isoformat())

            # 打印进度
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(
                    f"  迭代 {iteration+1:3d}/{n_opt}: "
                    f"样本={len(self.X_train):3d}, "
                    f"λ_t={lambda_t:.3f}, "
                    f"r_t={r_t:.3f}, "
                    f"得分={best_score:.4f}"
                )

        print(f"\n✓ 优化完成!")
        print(f"  - 完成 {n_opt} 次优化迭代")
        print(f"  - 最终样本数: {len(self.X_train)}")
        print(f"  - 最终 λ_t = {lambda_t:.3f}")
        print(f"  - 最终 r_t = {r_t:.3f}")

    def evaluate_model(self, n_test: int = 1000):
        """
        评估模型性能

        Parameters
        ----------
        n_test : int
            测试点数量
        """
        print("\n" + "=" * 80)
        print(f" 模型评估 (n_test={n_test}) ")
        print("=" * 80)

        # 生成测试点
        X_test = np.random.rand(n_test, 3)
        for i in range(3):
            X_test[:, i] = self.bounds[i, 0] + X_test[:, i] * (
                self.bounds[i, 1] - self.bounds[i, 0]
            )

        # 获取真实值(无噪声)
        y_test_true = np.array([self.subject.get_ground_truth_at(x) for x in X_test])

        # GP预测
        y_test_pred = self.acq_fn.gp_calculator.predict(X_test)

        # 计算指标
        mse = np.mean((y_test_true - y_test_pred) ** 2)
        mae = np.mean(np.abs(y_test_true - y_test_pred))
        rmse = np.sqrt(mse)

        # R²
        ss_res = np.sum((y_test_true - y_test_pred) ** 2)
        ss_tot = np.sum((y_test_true - np.mean(y_test_true)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # 相关系数
        corr = np.corrcoef(y_test_true, y_test_pred)[0, 1]

        print(f"\n测试集性能:")
        print(f"  - MSE  = {mse:.6f}")
        print(f"  - MAE  = {mae:.6f}")
        print(f"  - RMSE = {rmse:.6f}")
        print(f"  - R²   = {r2:.6f}")
        print(f"  - 相关系数 = {corr:.6f}")

        # 保存评估结果
        eval_results = {
            "n_test": n_test,
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "correlation": float(corr),
            "y_test_true": y_test_true.tolist(),
            "y_test_pred": y_test_pred.tolist(),
            "X_test": X_test.tolist(),
        }

        return eval_results

    def save_results(self, eval_results: dict = None):
        """保存实验结果"""
        print("\n" + "=" * 80)
        print(" 保存结果 ")
        print("=" * 80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存试次数据为CSV
        df = pd.DataFrame(self.history)
        csv_path = self.save_dir / f"experiment_data_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✓ 试次数据已保存: {csv_path}")

        # 2. 保存训练数据为NPZ
        npz_path = self.save_dir / f"training_data_{timestamp}.npz"
        np.savez(
            npz_path,
            X_train=self.X_train,
            y_train=self.y_train,
            param_names=self.param_names,
        )
        print(f"✓ 训练数据已保存: {npz_path}")

        # 3. 保存配置和元数据为JSON
        metadata = {
            "timestamp": timestamp,
            "n_trials": len(self.history["trial"]),
            "n_init": len([p for p in self.history["phase"] if p == "initialization"]),
            "n_opt": len([p for p in self.history["phase"] if p == "optimization"]),
            "subject_type": self.subject.true_function_type,
            "subject_noise": self.subject.noise_std,
            "acqf_params": {
                "lambda_min": self.acq_fn.lambda_min,
                "lambda_max": self.acq_fn.lambda_max,
                "tau_1": self.acq_fn.tau_1,
                "tau_2": self.acq_fn.tau_2,
                "gamma": self.acq_fn.gamma,
                "interaction_terms": self.acq_fn.interaction_terms,
            },
            "parameter_names": self.param_names,
            "bounds": self.bounds.tolist(),
            "seed": self.seed,
        }

        if eval_results is not None:
            metadata["evaluation"] = {
                "mse": eval_results["mse"],
                "mae": eval_results["mae"],
                "rmse": eval_results["rmse"],
                "r2": eval_results["r2"],
                "correlation": eval_results["correlation"],
            }

        json_path = self.save_dir / f"experiment_metadata_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ 元数据已保存: {json_path}")

        # 4. 保存完整评估结果
        if eval_results is not None:
            eval_path = self.save_dir / f"evaluation_results_{timestamp}.npz"
            np.savez(
                eval_path,
                X_test=np.array(eval_results["X_test"]),
                y_test_true=np.array(eval_results["y_test_true"]),
                y_test_pred=np.array(eval_results["y_test_pred"]),
                metrics={"mse": eval_results["mse"], "r2": eval_results["r2"]},
            )
            print(f"✓ 评估结果已保存: {eval_path}")

        return timestamp

    def visualize_results(self, timestamp: str, eval_results: dict = None):
        """生成可视化"""
        print("\n生成可视化...")

        df = pd.DataFrame(self.history)

        fig = plt.figure(figsize=(20, 12))

        # 1. Lambda和r_t演化
        ax1 = plt.subplot(3, 4, 1)
        opt_data = df[df["phase"] == "optimization"]
        if len(opt_data) > 0:
            ax1.plot(
                opt_data["trial"], opt_data["lambda_t"], "b-", linewidth=2, label="λ_t"
            )
            ax1.set_xlabel("Trial")
            ax1.set_ylabel("λ_t", color="b")
            ax1.tick_params(axis="y", labelcolor="b")
            ax1.grid(True, alpha=0.3)

            ax1_twin = ax1.twinx()
            ax1_twin.plot(
                opt_data["trial"], opt_data["r_t"], "r-", linewidth=2, label="r_t"
            )
            ax1_twin.set_ylabel("r_t", color="r")
            ax1_twin.tick_params(axis="y", labelcolor="r")
            ax1_twin.axhline(y=self.acq_fn.tau_1, color="r", linestyle="--", alpha=0.5)
            ax1_twin.axhline(y=self.acq_fn.tau_2, color="r", linestyle="--", alpha=0.5)
        ax1.set_title("Dynamic Weighting")

        # 2. 采集函数分数
        ax2 = plt.subplot(3, 4, 2)
        if len(opt_data) > 0:
            ax2.plot(opt_data["trial"], opt_data["acq_score"], "g-", linewidth=2)
            ax2.set_xlabel("Trial")
            ax2.set_ylabel("Acquisition Score")
            ax2.grid(True, alpha=0.3)
        ax2.set_title("Acquisition Function Score")

        # 3. 响应值演化
        ax3 = plt.subplot(3, 4, 3)
        init_data = df[df["phase"] == "initialization"]
        ax3.scatter(
            init_data["trial"],
            init_data["response"],
            c="blue",
            alpha=0.5,
            label="Init",
            s=30,
        )
        if len(opt_data) > 0:
            ax3.scatter(
                opt_data["trial"],
                opt_data["response"],
                c="red",
                alpha=0.5,
                label="Opt",
                s=30,
            )
        ax3.set_xlabel("Trial")
        ax3.set_ylabel("Response")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Response Over Time")

        # 4. 样本累积
        ax4 = plt.subplot(3, 4, 4)
        ax4.plot(df["trial"], range(1, len(df) + 1), "b-", linewidth=2)
        ax4.axvline(x=len(init_data), color="r", linestyle="--", label="Opt Start")
        ax4.set_xlabel("Trial")
        ax4.set_ylabel("Cumulative Samples")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Sample Growth")

        # 5. 参数空间覆盖 (x1 vs x2)
        ax5 = plt.subplot(3, 4, 5)
        scatter = ax5.scatter(
            init_data["x1"],
            init_data["x2"],
            c=init_data["response"],
            cmap="viridis",
            s=50,
            alpha=0.6,
            edgecolors="blue",
            linewidths=2,
            label="Init",
        )
        if len(opt_data) > 0:
            ax5.scatter(
                opt_data["x1"],
                opt_data["x2"],
                c=opt_data["response"],
                cmap="viridis",
                s=50,
                alpha=0.6,
                edgecolors="red",
                linewidths=2,
                label="Opt",
            )
        ax5.set_xlabel("Intensity")
        ax5.set_ylabel("Duration")
        ax5.set_title("Parameter Space (X1-X2)")
        plt.colorbar(scatter, ax=ax5, label="Response")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. 参数空间覆盖 (x2 vs x3)
        ax6 = plt.subplot(3, 4, 6)
        scatter = ax6.scatter(
            init_data["x2"],
            init_data["x3"],
            c=init_data["response"],
            cmap="viridis",
            s=50,
            alpha=0.6,
            edgecolors="blue",
            linewidths=2,
        )
        if len(opt_data) > 0:
            ax6.scatter(
                opt_data["x2"],
                opt_data["x3"],
                c=opt_data["response"],
                cmap="viridis",
                s=50,
                alpha=0.6,
                edgecolors="red",
                linewidths=2,
            )
        ax6.set_xlabel("Duration")
        ax6.set_ylabel("Noise Level")
        ax6.set_title("Parameter Space (X2-X3)")
        plt.colorbar(scatter, ax=ax6, label="Response")
        ax6.grid(True, alpha=0.3)

        # 7. 响应分布
        ax7 = plt.subplot(3, 4, 7)
        ax7.hist(df["response"], bins=30, alpha=0.7, edgecolor="black")
        ax7.set_xlabel("Response")
        ax7.set_ylabel("Frequency")
        ax7.set_title("Response Distribution")
        ax7.grid(True, alpha=0.3)

        # 8. 真实值 vs 观测值
        ax8 = plt.subplot(3, 4, 8)
        ax8.scatter(df["true_value"], df["response"], alpha=0.5, s=20)
        min_val = min(df["true_value"].min(), df["response"].min())
        max_val = max(df["true_value"].max(), df["response"].max())
        ax8.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        ax8.set_xlabel("True Value")
        ax8.set_ylabel("Observed Response")
        ax8.set_title("True vs Observed")
        ax8.grid(True, alpha=0.3)

        # 9-12: 评估结果(如果有)
        if eval_results is not None:
            y_true = np.array(eval_results["y_test_true"])
            y_pred = np.array(eval_results["y_test_pred"])

            # 9. 预测 vs 真实
            ax9 = plt.subplot(3, 4, 9)
            ax9.scatter(y_true, y_pred, alpha=0.3, s=10)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax9.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
            ax9.set_xlabel("True Value")
            ax9.set_ylabel("Predicted Value")
            ax9.set_title(f'Prediction (R²={eval_results["r2"]:.3f})')
            ax9.grid(True, alpha=0.3)

            # 10. 残差
            ax10 = plt.subplot(3, 4, 10)
            residuals = y_true - y_pred
            ax10.scatter(y_pred, residuals, alpha=0.3, s=10)
            ax10.axhline(y=0, color="r", linestyle="--", linewidth=2)
            ax10.set_xlabel("Predicted Value")
            ax10.set_ylabel("Residual")
            ax10.set_title("Residual Analysis")
            ax10.grid(True, alpha=0.3)

            # 11. 残差分布
            ax11 = plt.subplot(3, 4, 11)
            ax11.hist(residuals, bins=50, alpha=0.7, edgecolor="black")
            ax11.set_xlabel("Residual")
            ax11.set_ylabel("Frequency")
            ax11.set_title("Residual Distribution")
            ax11.axvline(x=0, color="r", linestyle="--", linewidth=2)
            ax11.grid(True, alpha=0.3)

            # 12. 误差统计
            ax12 = plt.subplot(3, 4, 12)
            ax12.axis("off")
            stats_text = f"""
Model Performance

MSE:  {eval_results['mse']:.4f}
MAE:  {eval_results['mae']:.4f}
RMSE: {eval_results['rmse']:.4f}
R²:   {eval_results['r2']:.4f}
Corr: {eval_results['correlation']:.4f}

Training Samples: {len(self.X_train)}
Test Samples: {eval_results['n_test']}
"""
            ax12.text(
                0.1,
                0.5,
                stats_text,
                fontsize=12,
                family="monospace",
                verticalalignment="center",
            )

        plt.tight_layout()

        # 保存图像
        fig_path = self.save_dir / f"experiment_results_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"✓ 可视化已保存: {fig_path}")
        plt.close()


def main():
    """运行完整实验"""
    print("=" * 80)
    print(" 完整模拟实验 - 使用VarianceReductionWithCoverageAcqf ")
    print("=" * 80)

    # 创建实验
    exp = ExperimentRunner(
        config_path="experiment_config.ini",
        subject_type="nonlinear_interaction",  # 最接近真实心理物理函数
        subject_noise=0.3,
        save_dir="results",
        seed=42,
    )

    # 运行实验
    exp.run_initialization_phase(n_init=20)
    exp.run_optimization_phase(n_opt=40, n_candidates=300)

    # 评估模型
    eval_results = exp.evaluate_model(n_test=1000)

    # 保存结果
    timestamp = exp.save_results(eval_results)

    # 生成可视化
    exp.visualize_results(timestamp, eval_results)

    print("\n" + "=" * 80)
    print(" 实验完成! ")
    print("=" * 80)
    print(f"\n所有结果已保存到: {exp.save_dir}")
    print(f"  - 试次数据: experiment_data_{timestamp}.csv")
    print(f"  - 训练数据: training_data_{timestamp}.npz")
    print(f"  - 元数据: experiment_metadata_{timestamp}.json")
    print(f"  - 评估结果: evaluation_results_{timestamp}.npz")
    print(f"  - 可视化: experiment_results_{timestamp}.png")


if __name__ == "__main__":
    main()

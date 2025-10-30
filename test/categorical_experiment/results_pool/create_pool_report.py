#!/usr/bin/env python3
"""
Generate a Markdown report with key visualizations for the latest pool-based EUR(V4) run.

Inputs: auto-detect the latest pool_metrics_*.json and matching pool_trials_*.csv in this folder.
Outputs: PNG figures and REPORT_<timestamp>.md in the same folder.
"""

import os
import re
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    # Prefer CJK-capable fonts on Windows; provide fallbacks
    from matplotlib import rcParams

    rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Segoe UI Symbol",
        "Arial",
    ]
    rcParams["axes.unicode_minus"] = False
except Exception as e:
    plt = None


HERE = os.path.dirname(__file__)


def find_latest_results() -> Tuple[str, str, str]:
    files = os.listdir(HERE)
    metrics = [
        f for f in files if f.startswith("pool_metrics_") and f.endswith(".json")
    ]
    if not metrics:
        raise FileNotFoundError("No pool_metrics_*.json found in results_pool.")

    # extract timestamp
    def ts_of(name: str) -> str:
        m = re.search(r"(\d{8}_\d{6})", name)
        return m.group(1) if m else ""

    metrics.sort(key=lambda x: ts_of(x))
    latest_metrics = metrics[-1]
    ts = ts_of(latest_metrics)
    trials = f"pool_trials_{ts}.csv"
    return os.path.join(HERE, latest_metrics), os.path.join(HERE, trials), ts


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trials_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def fig_trajectories(summary: Dict, out_png: str) -> Optional[str]:
    if plt is None:
        return None
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.ravel()
    # 1: lambda
    axes[0].plot(summary.get("lambda_traj", []), color="tab:blue")
    axes[0].set_title("λ_t 轨迹")
    axes[0].set_xlabel("iter")
    # 2: r_t
    axes[1].plot(summary.get("r_t_traj", []), color="tab:orange")
    axes[1].set_title("r_t 轨迹 (主效应方差比)")
    axes[1].set_xlabel("iter")
    # 3: main var
    axes[2].plot(summary.get("main_var_traj", []), color="tab:green")
    axes[2].set_title("主效应后验方差均值")
    axes[2].set_xlabel("iter")
    # 4: inter var
    axes[3].plot(summary.get("inter_var_traj", []), color="tab:red")
    axes[3].set_title("交互项后验方差均值")
    axes[3].set_xlabel("iter")
    # 5: dvar main
    axes[4].plot(summary.get("dvar_main_traj", []), color="tab:purple")
    axes[4].set_title("ΔVar_main 平均 (选点前预期)")
    axes[4].set_xlabel("iter")
    # 6: dvar inter
    axes[5].plot(summary.get("dvar_inter_traj", []), color="tab:brown")
    axes[5].set_title("ΔVar_inter 平均 (选点前预期)")
    axes[5].set_xlabel("iter")
    fig.suptitle("EUR(V4) 学习度轨迹", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def fig_coverage(summary: Dict, out_png: str) -> Optional[str]:
    if plt is None:
        return None
    # per-variable coverage
    pv = summary.get("per_variable_coverage", {})
    names = list(pv.keys())
    covs = [(pv[k]["coverage"] if k in pv else 0.0) for k in names]
    # pairwise coverage
    pc = summary.get("pairwise_coverage", {})
    pairs = list(pc.keys())
    pair_covs = [(pc[k]["coverage"] if k in pc else 0.0) for k in pairs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(names, [c * 100 for c in covs], color="teal")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("各变量层级覆盖率 (%)")
    axes[0].tick_params(axis="x", rotation=20)

    # sort pairwise by coverage desc for readability
    order = np.argsort(pair_covs)[::-1]
    pairs_sorted = [pairs[i] for i in order]
    pair_covs_sorted = [pair_covs[i] * 100 for i in order]
    axes[1].barh(pairs_sorted, pair_covs_sorted, color="gold")
    axes[1].set_xlim(0, 105)
    axes[1].set_title("成对组合覆盖率 (%)")

    fig.suptitle("覆盖度概览", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def fig_ratings(trials: List[Dict[str, str]], out_png: str) -> Optional[str]:
    if plt is None:
        return None
    # Extract sequences
    try:
        trial_idx = [int(r["trial"]) for r in trials]
        rating = [float(r["rating"]) for r in trials]
        true_score = [float(r.get("true_score", np.nan)) for r in trials]
    except Exception:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # timeline
    axes[0].plot(
        trial_idx, rating, label="rating (observed)", color="tab:blue", alpha=0.7
    )
    if not np.all(np.isnan(true_score)):
        axes[0].plot(
            trial_idx, true_score, label="true_score", color="tab:orange", alpha=0.7
        )
    axes[0].set_title("评分随试次变化")
    axes[0].set_xlabel("trial")
    axes[0].set_ylabel("score")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # histogram
    axes[1].hist(rating, bins=np.arange(0.5, 10.6, 1.0), edgecolor="black", alpha=0.7)
    axes[1].set_title("评分分布 (Likert 1-10)")
    axes[1].set_xlabel("rating")
    axes[1].set_ylabel("count")
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("评分概览", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def write_report_md(
    summary: Dict, figs: Dict[str, Optional[str]], out_md: str, ts: str
) -> str:
    lines: List[str] = []
    lines.append(f"# Pool-based EUR(V4) 报告 ({ts})\n")
    cfg = summary.get("config", {})
    lines.append("## 配置\n")
    lines.append(f"- 预算: {cfg.get('n_init', '?')} init + {cfg.get('n_opt', '?')} opt")
    lines.append(
        f"- 用户类型: {cfg.get('user_type', '?')}  噪声: {cfg.get('noise', '?')}"
    )
    lines.append(
        f"- γ={cfg.get('gamma','?')}  λ∈[{cfg.get('lambda_min','?')}, {cfg.get('lambda_max','?')}]  τ1={cfg.get('tau1','?')}  τ2={cfg.get('tau2','?')}"
    )
    lines.append("")

    lines.append("## 关键指标\n")
    lines.append(
        f"- 总试验数: {summary.get('total_trials', '?')}  唯一设计: {summary.get('unique_designs', '?')}  去重率: {summary.get('unique_rate', 0)*100:.1f}%"
    )
    lines.append(
        f"- 平均评分: {summary.get('mean_rating', float('nan')):.3f}  最高评分: {summary.get('max_rating', float('nan')):.2f}"
    )
    lines.append("")

    # per-variable coverage
    lines.append("### 变量覆盖率\n")
    pv = summary.get("per_variable_coverage", {})
    for k, v in pv.items():
        lines.append(
            f"- {k}: {v['coverage']*100:.1f}% ({int(v['seen'])}/{int(v['total'])})"
        )
    lines.append("")

    # pairwise coverage (top few)
    lines.append("### 成对组合覆盖率 (Top)\n")
    pc = summary.get("pairwise_coverage", {})
    pairs = sorted(pc.items(), key=lambda kv: kv[1]["coverage"], reverse=True)
    for k, v in pairs[:6]:
        lines.append(
            f"- {k}: {v['coverage']*100:.1f}% ({int(v['seen'])}/{int(v['total'])})"
        )
    lines.append("")

    # figures
    if any(figs.values()):
        lines.append("## 可视化\n")
        if figs.get("traj"):
            lines.append(f"![学习度轨迹]({os.path.basename(figs['traj'])})\n")
        if figs.get("cov"):
            lines.append(f"![覆盖度概览]({os.path.basename(figs['cov'])})\n")
        if figs.get("rating"):
            lines.append(f"![评分概览]({os.path.basename(figs['rating'])})\n")

    # evaluation text
    lines.append("## 评价\n")
    unique_rate = summary.get("unique_rate", 0)
    eval_lines = []
    if unique_rate >= 0.99:
        eval_lines.append("结构性去重达成 (≈100%)，重复采样已被有效避免。")
    pv_ok = all(v.get("coverage", 0) >= 0.9 for v in pv.values())
    if pv_ok:
        eval_lines.append("各变量层级覆盖充分，有助于稳健估计主效应。")
    # pairwise coverage heuristic
    pc_ok = all(v.get("coverage", 0) >= 0.9 for v in pc.values()) if pc else False
    if pc_ok:
        eval_lines.append("成对交互覆盖充分，二级交互可较为可靠地学习与验证。")
    # lambda trajectory comment
    lam = summary.get("lambda_traj", [])
    if lam and lam[-1] > lam[0]:
        eval_lines.append(
            "λ_t 随试次上升，表明主效应收敛后逐步加大交互项权重，策略符合预期。"
        )
    # variance trends
    mv = summary.get("main_var_traj", [])
    iv = summary.get("inter_var_traj", [])
    if mv and iv and (mv[-1] < mv[0]) and (iv[-1] < iv[0]):
        eval_lines.append(
            "主/交互后验方差整体下降，模型逐步收敛，信息增益递减符合学习规律。"
        )
    if not eval_lines:
        eval_lines.append(
            "运行完成。建议结合图像与指标进一步检视探索-利用权衡与参数设置。"
        )
    for l in eval_lines:
        lines.append(f"- {l}")

    # write
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_md


def main():
    metrics_path, trials_path, ts = find_latest_results()
    summary = load_json(metrics_path)
    trials = load_trials_csv(trials_path)

    figs: Dict[str, Optional[str]] = {"traj": None, "cov": None, "rating": None}
    # figures
    figs["traj"] = fig_trajectories(
        summary, os.path.join(HERE, f"pool_fig_trajectories_{ts}.png")
    )
    figs["cov"] = fig_coverage(
        summary, os.path.join(HERE, f"pool_fig_coverage_{ts}.png")
    )
    figs["rating"] = fig_ratings(
        trials, os.path.join(HERE, f"pool_fig_ratings_{ts}.png")
    )

    # report
    md_path = os.path.join(HERE, f"REPORT_{ts}.md")
    out_md = write_report_md(summary, figs, md_path, ts)

    print("Report generated:", out_md)
    for k, v in figs.items():
        if v:
            print(f"Figure [{k}]: {v}")
        else:
            print(f"Figure [{k}]: skipped (matplotlib not available)")


if __name__ == "__main__":
    main()

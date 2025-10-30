#!/usr/bin/env python3
"""
Pool-based EUR(V4) 模拟实验（结构性去重 + 主/交互效应学习度指标）

要点：
- 候选池 = 全部 360 组合（色彩×布局×字体×动画），生成器严格去重；
- 模型：AEPsych GPRegressionModel(dim=4)（将离散维规范化为数值）;
- 采集策略：EURAcqfV4（α_info + α_cov，动态 λ_t），通过 PoolBasedGenerator 在池上打分拣选；
- 预算：15 初始化 + 45 优化 = 60；
- 报告：唯一率、变量层级覆盖率、成对交互覆盖率、λ_t 轨迹、主/交互参数方差轨迹、所选点的 ΔVar_main/inter 均值轨迹；
- 输出：CSV（trial 明细）、JSON（指标）、可选控制台摘要；

运行：python run_pool_based_experiment.py --acqf v4 --n-init 15 --n-opt 45
"""

import os
import sys
import argparse
import json
from datetime import datetime
from itertools import product, combinations
from typing import Dict, List, Tuple

import numpy as np
import torch

# 路径：加入根与 temp_aepsych 以便直接 import
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "temp_aepsych"))
sys.path.insert(0, ROOT)

from aepsych.models import GPRegressionModel
from extensions.custom_generators.pool_based_generator import PoolBasedGenerator
from extensions.dynamic_eur_acquisition import EURAcqfV4
from extensions.dynamic_eur_acquisition.test.categorical_experiment.virtual_user import (
    VirtualUser,
)


# ------------------------------ 设计空间编码 ------------------------------
COLORS = VirtualUser.COLOR_SCHEMES  # 5
LAYOUTS = VirtualUser.LAYOUTS  # 4
FONTS = VirtualUser.FONT_SIZES  # [12,14,16,18,20,22] 共6级 → 总空间=5*4*6*3=360
ANIMS = VirtualUser.ANIMATIONS  # 3


def encode_design(design: Dict) -> np.ndarray:
    """将设计 dict 编码为数值向量 [c, l, f, a]（float64）。"""
    c = COLORS.index(design["color_scheme"])  # 0..4
    l = LAYOUTS.index(design["layout"])  # 0..3
    f = float(design["font_size"])  # 12..22（整数，但模型端为浮点）
    a = ANIMS.index(design["animation"])  # 0..2
    return np.array([c, l, f, a], dtype=np.float64)


def decode_vector(x: np.ndarray) -> Dict:
    """将向量解码为设计 dict（用于记录）。"""
    c = COLORS[int(round(x[0]))]
    l = LAYOUTS[int(round(x[1]))]
    f = int(round(x[2]))
    a = ANIMS[int(round(x[3]))]
    return {
        "color_scheme": c,
        "layout": l,
        "font_size": f,
        "animation": a,
    }


def build_full_pool() -> torch.Tensor:
    """生成 360 组合的完整候选池（float64）。"""
    all_designs = []
    for c, l, f, a in product(COLORS, LAYOUTS, FONTS, ANIMS):
        all_designs.append(
            {
                "color_scheme": c,
                "layout": l,
                "font_size": f,
                "animation": a,
            }
        )
    X = np.stack([encode_design(d) for d in all_designs], axis=0)
    return torch.tensor(X, dtype=torch.float64)


# ------------------------------ 统计/指标工具 ------------------------------
def coverage_per_variable(trials: List[Dict]) -> Dict[str, Dict[str, float]]:
    """计算各变量层级覆盖率（已出现/总层级）。"""
    vars_levels = {
        "color_scheme": set(COLORS),
        "layout": set(LAYOUTS),
        "font_size": set(FONTS),
        "animation": set(ANIMS),
    }
    seen = {k: set() for k in vars_levels}
    for row in trials:
        for k in seen:
            seen[k].add(row[k])
    out = {}
    for k, universe in vars_levels.items():
        out[k] = {
            "seen": float(len(seen[k])),
            "total": float(len(universe)),
            "coverage": float(len(seen[k])) / float(len(universe)),
        }
    return out


def pairwise_coverage(trials: List[Dict]) -> Dict[str, Dict[str, float]]:
    """计算成对变量组合覆盖率（已出现/总组合）。"""
    var_names = ["color_scheme", "layout", "font_size", "animation"]
    domains = {
        "color_scheme": set(COLORS),
        "layout": set(LAYOUTS),
        "font_size": set(FONTS),
        "animation": set(ANIMS),
    }
    out: Dict[str, Dict[str, float]] = {}
    for v1, v2 in combinations(var_names, 2):
        key = f"{v1}×{v2}"
        seen = set((row[v1], row[v2]) for row in trials)
        total = len(domains[v1]) * len(domains[v2])
        out[key] = {
            "seen": float(len(seen)),
            "total": float(total),
            "coverage": float(len(seen)) / float(total),
        }
    return out


def summarize_variance(acqf: EURAcqfV4) -> Dict[str, float]:
    """读取当前主/交互参数方差（均值），以及 λ_t、r_t。"""
    # 刷新 Dt
    acqf._ensure_fresh_data()
    var_cur = acqf._var_current
    var_ini = acqf._var_initial
    if var_cur is None or var_ini is None:
        return {"lambda_t": float(acqf._compute_dynamic_lambda())}
    off = 1 if acqf.gp_calculator.include_intercept else 0
    n_main = acqf._n_features or 0
    main_cur = var_cur[off : off + n_main]
    main_ini = var_ini[off : off + n_main]
    main_ratio = (
        np.mean(main_cur / np.clip(main_ini, 1e-10, None)) if n_main > 0 else 1.0
    )
    inter_cur = var_cur[off + n_main :]
    d = {
        "lambda_t": float(acqf._compute_dynamic_lambda()),
        "r_t": float(main_ratio),
        "main_var_mean": float(np.mean(main_cur)) if n_main > 0 else 0.0,
        "inter_var_mean": float(np.mean(inter_cur)) if len(inter_cur) > 0 else 0.0,
    }
    return d


def compute_delta_var_for_point(acqf: EURAcqfV4, x: np.ndarray) -> Dict[str, float]:
    """计算单点的 ΔVar 主/交互均值（使用当前 gp_calculator，未真正加入该点）。"""
    acqf._ensure_fresh_data()
    main_red, inter_red = acqf.gp_calculator.compute_variance_reduction(
        x.reshape(1, -1)
    )
    main_red = np.maximum(main_red, 0.0) if main_red is not None else np.array([])
    inter_red = np.maximum(inter_red, 0.0) if inter_red is not None else np.array([])
    return {
        "dvar_main_mean": float(np.mean(main_red)) if len(main_red) > 0 else 0.0,
        "dvar_inter_mean": float(np.mean(inter_red)) if len(inter_red) > 0 else 0.0,
    }


# ------------------------------ 主流程 ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Pool-based EUR(V4) 模拟实验")
    p.add_argument("--n-init", type=int, default=15, help="初始化样本数")
    p.add_argument("--n-opt", type=int, default=45, help="优化迭代数")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument(
        "--user-type",
        type=str,
        default="balanced",
        choices=["balanced", "minimalist", "colorful"],
        help="虚拟用户类型",
    )
    p.add_argument("--noise", type=float, default=0.5, help="评分噪声水平")
    p.add_argument("--gamma", type=float, default=0.5, help="覆盖项系数 γ")
    p.add_argument("--lambda-min", type=float, default=0.2)
    p.add_argument("--lambda-max", type=float, default=2.0)
    p.add_argument("--tau1", type=float, default=0.5)
    p.add_argument("--tau2", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 构建池与边界
    pool = build_full_pool()  # [360,4]
    lb = torch.tensor([0.0, 0.0, 12.0, 0.0], dtype=torch.float64)
    ub = torch.tensor([4.0, 3.0, 22.0, 2.0], dtype=torch.float64)

    # 模型
    model = GPRegressionModel(dim=4)

    # 虚拟用户
    user = VirtualUser(user_type=args.user_type, noise_level=args.noise, seed=seed)

    # 初始化：随机取样 n_init 个池内点
    init_idx = torch.randperm(pool.shape[0])[: args.n_init]
    X_init = pool[init_idx]
    trials: List[Dict] = []
    y_init = []
    for i, x in enumerate(X_init):
        design = decode_vector(x.cpu().numpy())
        obs = user.rate_design(design)  # 1..10 离散评分
        y_init.append(float(obs["rating"]))
        trials.append(
            {
                "trial": i + 1,
                "phase": "initialization",
                **design,
                "rating": float(obs["rating"]),
                "true_score": float(user.get_ground_truth(design)),
                "rt": float(obs["rt"]),
            }
        )
    y_init = torch.tensor(y_init, dtype=torch.float64)

    # 拟合模型
    model.fit(X_init, y_init)

    # 构造 EUR V4 与 PoolBasedGenerator
    acqf = EURAcqfV4(
        model=model,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        tau_1=args.tau1,
        tau_2=args.tau2,
        gamma=args.gamma,
        # 交互项：此处默认“所有成对”可在 gp_variance 中按索引生成；若为空则由 GPVarianceCalculator 不添加交互项
        # 也可显式传 [] 仅学习主效应
        interaction_terms=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        debug_components=False,
    )

    gen = PoolBasedGenerator(
        lb=lb,
        ub=ub,
        pool_points=pool,
        acqf=EURAcqfV4,
        acqf_kwargs={
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
            "tau_1": args.tau1,
            "tau_2": args.tau2,
            "gamma": args.gamma,
            "interaction_terms": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        },
        allow_resampling=False,
        shuffle=False,
        seed=seed,
    )

    # 迭代优化
    X_all = [x for x in X_init]
    y_all = [float(v) for v in y_init]
    metrics_traj = (
        []
    )  # 每步：lambda_t, r_t, main_var_mean, inter_var_mean, dvar_main_mean, dvar_inter_mean

    for t in range(args.n_opt):
        # 选点（去重由生成器保证）
        x_next = gen.gen(num_points=1, model=model)[0].detach().cpu().numpy()

        # 记录 ΔVar（选点前的预期降低）
        dvar = compute_delta_var_for_point(acqf, x_next)
        vstat = summarize_variance(acqf)
        metrics_traj.append({**vstat, **dvar, "iter": t + 1})

        # 打分并更新
        design = decode_vector(x_next)
        obs = user.rate_design(design)
        y_obs = float(obs["rating"])

        X_all.append(torch.tensor(x_next, dtype=torch.float64))
        y_all.append(y_obs)
        model.update(torch.stack(X_all), torch.tensor(y_all, dtype=torch.float64))

        trials.append(
            {
                "trial": len(trials) + 1,
                "phase": "optimization",
                **design,
                "rating": y_obs,
                "true_score": float(user.get_ground_truth(design)),
                "rt": float(obs["rt"]),
            }
        )

    # 汇总与输出
    total = len(trials)
    uniq = len(
        {
            (r["color_scheme"], r["layout"], r["font_size"], r["animation"])
            for r in trials
        }
    )
    ratings = np.array([r["rating"] for r in trials], dtype=float)

    cov_var = coverage_per_variable(trials)
    cov_pair = pairwise_coverage(trials)

    summary = {
        "total_trials": total,
        "unique_designs": uniq,
        "unique_rate": uniq / float(total),
        "mean_rating": float(ratings.mean()),
        "max_rating": float(ratings.max()),
        "per_variable_coverage": cov_var,
        "pairwise_coverage": cov_pair,
        "lambda_traj": [m.get("lambda_t") for m in metrics_traj],
        "r_t_traj": [m.get("r_t") for m in metrics_traj],
        "main_var_traj": [m.get("main_var_mean") for m in metrics_traj],
        "inter_var_traj": [m.get("inter_var_mean") for m in metrics_traj],
        "dvar_main_traj": [m.get("dvar_main_mean") for m in metrics_traj],
        "dvar_inter_traj": [m.get("dvar_inter_mean") for m in metrics_traj],
        "config": {
            "n_init": args.n_init,
            "n_opt": args.n_opt,
            "user_type": args.user_type,
            "noise": args.noise,
            "gamma": args.gamma,
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
            "tau1": args.tau1,
            "tau2": args.tau2,
            "seed": seed,
        },
    }

    # 输出到 results_pool 目录
    out_dir = os.path.join(os.path.dirname(__file__), "results_pool")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # trials CSV
    import csv

    csv_path = os.path.join(out_dir, f"pool_trials_{ts}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial",
                "phase",
                "color_scheme",
                "layout",
                "font_size",
                "animation",
                "rating",
                "true_score",
                "rt",
            ],
        )
        writer.writeheader()
        for row in trials:
            writer.writerow(row)

    # metrics JSON
    json_path = os.path.join(out_dir, f"pool_metrics_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print("Pool-based EUR(V4) Run Complete")
    print(f"Total trials: {total}; Unique: {uniq} ({summary['unique_rate']*100:.1f}%)")
    print(
        f"Mean rating: {summary['mean_rating']:.3f}; Max rating: {summary['max_rating']:.2f}"
    )
    print("Per-variable coverage (%):")
    for k, v in cov_var.items():
        print(f"  - {k}: {v['coverage']*100:.1f}% ({int(v['seen'])}/{int(v['total'])})")
    print("Pairwise coverage (top 3):")
    top_pairs = sorted(
        cov_pair.items(), key=lambda kv: kv[1]["coverage"], reverse=True
    )[:3]
    for k, v in top_pairs:
        print(f"  - {k}: {v['coverage']*100:.1f}% ({int(v['seen'])}/{int(v['total'])})")
    print(f"Artifacts: {csv_path} , {json_path}")


if __name__ == "__main__":
    main()

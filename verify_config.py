"""验证动态权重配置是否适合你的采样预算"""

def verify_gamma_trajectory(
    total_budget: int,
    tau_n_min: int = 3,
    tau_n_max: int = 40,
    gamma_max: float = 0.5,
    gamma_min: float = 0.1,
):
    """
    验证γ_t轨迹是否适合你的采样预算

    Args:
        total_budget: 你的总采样次数（如20、30、50等）
        tau_n_min, tau_n_max, gamma_max, gamma_min: 当前配置
    """
    print("=" * 70)
    print(f"采样预算：{total_budget} 次")
    print(f"配置：tau_n_min={tau_n_min}, tau_n_max={tau_n_max}")
    print(f"      gamma_max={gamma_max}, gamma_min={gamma_min}")
    print("=" * 70)

    def compute_gamma_t(n):
        if n < tau_n_min:
            return gamma_max
        elif n >= tau_n_max:
            return gamma_min
        else:
            t_ratio = (n - tau_n_min) / (tau_n_max - tau_n_min)
            return gamma_max - (gamma_max - gamma_min) * t_ratio

    # 分析关键阶段
    milestones = [1, 5, 10, 15, 20, 25, 30, 40, 50]
    milestones = [n for n in milestones if n <= total_budget]
    milestones.append(total_budget)

    print("\nγ_t 轨迹分析：")
    print("-" * 70)
    print(f"{'样本数':>6} | {'γ_t':>6} | {'信息权重':>10} | {'覆盖权重':>10} | 阶段")
    print("-" * 70)

    for n in sorted(set(milestones)):
        gamma = compute_gamma_t(n)
        info_weight = (1 - gamma) * 100
        cov_weight = gamma * 100

        # 判断阶段
        if gamma > 0.4:
            phase = "探索"
        elif gamma > 0.2:
            phase = "平衡"
        else:
            phase = "精细化"

        print(f"{n:6d} | {gamma:6.3f} | {info_weight:9.1f}% | {cov_weight:9.1f}% | {phase}")

    # 分析建议
    final_gamma = compute_gamma_t(total_budget)
    print("-" * 70)

    if final_gamma > 0.3:
        print("⚠️ 警告：实验结束时覆盖权重仍>30%")
        print(f"   当前设置下，{total_budget}次采样时 γ_t={final_gamma:.3f}")
        print(f"   建议调整：tau_n_max = {int(total_budget * 0.7)} （约预算的70%）")
        print(f"   效果：实验结束时 γ_t ≈ {compute_gamma_t_new(total_budget, total_budget * 0.7):.3f}")
        return False
    elif final_gamma < 0.15:
        print("✅ 配置合理：实验后期充分聚焦信息项")
        return True
    else:
        print("✅ 配置合理：平衡探索与精细化")
        return True

def compute_gamma_t_new(n, tau_n_max_new, tau_n_min=3, gamma_max=0.5, gamma_min=0.1):
    """计算调整后的gamma"""
    if n < tau_n_min:
        return gamma_max
    elif n >= tau_n_max_new:
        return gamma_min
    else:
        t_ratio = (n - tau_n_min) / (tau_n_max_new - tau_n_min)
        return gamma_max - (gamma_max - gamma_min) * t_ratio

# ===== 测试不同预算场景 =====
if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# 验证你的实验配置")
    print("#" * 70)

    # 场景1：小预算实验
    print("\n【场景1：小预算实验（20次采样）】")
    verify_gamma_trajectory(
        total_budget=20,
        tau_n_max=40,  # 原始默认值
    )

    print("\n" + "="*70)
    print("建议调整后：")
    verify_gamma_trajectory(
        total_budget=20,
        tau_n_max=20,  # 调整为预算的100%
        gamma_min=0.05,
    )

    # 场景2：中等预算实验
    print("\n\n【场景2：中等预算实验（30次采样）】")
    verify_gamma_trajectory(
        total_budget=30,
        tau_n_max=40,  # 原始默认值
    )

    print("\n" + "="*70)
    print("建议调整后：")
    verify_gamma_trajectory(
        total_budget=30,
        tau_n_max=25,  # 调整为预算的80%
    )

    # 场景3：充足预算实验
    print("\n\n【场景3：充足预算实验（50次采样）】")
    verify_gamma_trajectory(
        total_budget=50,
        tau_n_max=40,  # 原始默认值
    )

    print("\n\n" + "#" * 70)
    print("# 总结")
    print("#" * 70)
    print("""
推荐配置公式：
  tau_n_max = total_budget × 0.7  （预算的70%处转向精细化）
  tau_n_min = 3                   （前3次采样全局探索）
  gamma_max = 0.5                 （早期50%覆盖权重）
  gamma_min = 0.05-0.1            （后期5-10%覆盖权重）

示例：
  - 20次预算 → tau_n_max=15
  - 30次预算 → tau_n_max=20-25
  - 50次预算 → tau_n_max=35-40
    """)

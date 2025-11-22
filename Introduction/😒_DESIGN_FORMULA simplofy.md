# Dynamic EUR Acquisition Function

## 设计意图

**目标场景**：心理学实验中的有限预算探索

- **问题**：设计空间庞大（中高维），被试精力有限，无法遍历
- **目标 1**：通过贝叶斯模型估计主效应与交互效应（用于统计推断）
- **目标 2**：构建预测模型（泛化到未观测区域）
- **核心挑战**：在 20-40 次采样内同时实现效应发现与模型构建

**设计策略**：ANOVA分解 + 动态权重 + 空间覆盖

- 早期：主效应探索 + 空间覆盖（发现重要因素，建立全局模型）
- 后期：交互探索 + 精细化采样（发现复杂关系，提升预测精度）

---

## 核心公式

$$
\alpha(x) = \text{info}(x) + \gamma_t \cdot \text{cov}(x)
$$

---

## 1. 信息项：ANOVA 分解

$$
\text{info}(x) = \frac{1}{d}\sum_{j=1}^{d} \Delta_j + \lambda_t(r_t) \cdot \frac{1}{|P|}\sum_{(i,j) \in P} \Delta_{ij}
$$

**效应定义**：

- 主效应：$\Delta_j = I(x_j) - I(x)$
- 二阶交互：$\Delta_{ij} = I(x_{ij}) - I(x_i) - I(x_j) + I(x)$
- 三阶交互（Multi 版）：$\Delta_{ijk} = I(x_{ijk}) - I(x_{ij}) - I(x_{ik}) - I(x_{jk}) + I(x_i) + I(x_j) + I(x_k) - I(x)$

**信息度量** $I(x)$：序数模型用熵 $H[p(y|x)]$，回归模型用方差 $\text{Var}[f(x)]$

## 2. 覆盖项

$$
\text{cov}(x) = \min_i \text{dist}_{\text{Gower}}(x, x_i)
$$

---

## 3. 动态权重

### λₜ：交互探索自适应

参数收敛度 $r_t = \frac{1}{d}\sum_j \frac{\text{Var}[\theta_j|D_t]}{\text{Var}[\theta_j|D_0]}$ 驱动分段函数：

$$
\lambda_t(r_t) = \begin{cases}
\lambda_{\min} & r_t > \tau_1 \text{ (未收敛→主效应)} \\
\text{线性插值} & \tau_2 \le r_t \le \tau_1 \\
\lambda_{\max} & r_t < \tau_2 \text{ (已收敛→交互)}
\end{cases}
$$

### γₜ：覆盖自适应

$$
\gamma_t = \underbrace{\text{线性}(n; \gamma_{\max} \to \gamma_{\min})}_{\text{样本数}} \times \underbrace{\{0.8, 1.0, 1.2\}}_{\text{参数收敛微调}}
$$

---

## 4. 算法流程

```python
# 批量构造 + 一次评估（~20x 加速）
I_base = metric(X)
X_batch = cat([local_sample(X, [j]) for j in range(d)],      # 主效应
              [local_sample(X, [i,j]) for (i,j) in pairs])   # 交互
I_batch = metric(X_batch)  # 关键：1 次模型调用

# ANOVA 分解（不确定性导向：Δ>0）
delta_main = [clamp(I_j - I_base, 0) for I_j in I_batch[:d]]
delta_pair = [clamp(I_ij - I_i - I_j + I_base, 0) for (i,j) in pairs]

# 融合
info = mean(delta_main) + compute_lambda(r_t) * mean(delta_pair)
alpha = norm(info) + compute_gamma(n, r_t) * norm(gower_distance(X))
```

---

## 5. 关键特性

| 特性       | 实现                                            |
| -------- | --------------------------------------------- |
| **策略**   | 不确定性导向（$\Delta > 0$）：探索未知效应                   |
| **自适应**  | $\lambda_t(r_t), \gamma_t(n, r_t)$：早期主效应，后期交互 |
| **混合变量** | 分类（离散采样）/ 整数（舍入）/ 连续（高斯）                      |
| **批量优化** | 1 次模型调用，~20x 加速                               |
| **版本**   | Pair（二阶）/ Multi（任意阶，ANOVA 容斥）                 |

---

## 6. 实验设计方略与数据使用

**定位**：GP 仅用于自适应采样(选 x)，不参与最终效应/统计显著性分析；最终推断只基于原始观测数据。

**每被试独立 GP**：

- 优点：捕捉个体差异；避免跨被试信息泄漏；保持后续混合/层次模型独立性；调度简单。
- 风险：单被试样本少(20–40)→超参数估计不稳定；重复训练成本增加。
- 适用前提：关注个体层解释或个体协方差差异大；不追求全局统一最优点。

**改进选项（需要时）**：

1) 共享初始化超参数/经验先验（长度尺度、噪声）稳定早期；
2) 若发现跨被试结构稳定 → 层次/多任务 GP (ICM/LMC) 局部共享；
3) 交互对可先全局筛选，再在各被试内细化。

**结论**：当前策略在“个体差异显著 + 双目标探索”场景下合理；若目标转向“群体泛化”应考虑部分共享结构。

---

## 7. 适用条件

### 核心判据：采样密度

$$
\rho = \frac{n_{\text{samples}}}{\text{design space size}} < 0.1
$$

| 场景        | 维度   | 预算    | 配置                            |
| --------- | ---- | ----- | ----------------------------- |
| 小规模       | 2-3  | 10-20 | `tau_n_max=10, gamma_max=0.6` |
| 中等（心理学典型） | 4-6  | 20-40 | `tau_n_max=25, gamma_max=0.5` |
| 大规模       | 7-10 | 40-60 | `tau_n_max=35, gamma_max=0.4` |

**自动配置**：提供 `total_budget` 时，`tau_n_max = 0.7×budget`, `gamma_min = 0.05(budget<30) else 0.1`

### 何时使用 ✅

**必备条件**（全部满足）：

1. **稀疏采样**：$\rho < 0.1$（如 6 维×3⁶ 空间，<73 次采样）
2. **探索性目标**：不知道哪些效应存在，需要发现
3. **双重需求**：既要效应估计（ANOVA），又要预测模型

**增强适配**（越多越好）：

- 混合变量类型（分类/整数/连续）
- 序数响应（Likert 量表）
- 被试个体差异大（需混合模型）
- 有理论指导的交互假设（可预设交互对）

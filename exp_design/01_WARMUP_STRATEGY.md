# 预热阶段方案详解

## 🎯 预热阶段的目标

**不是**：训练"最终模型"（会导致过拟合）
**而是**：Meta-learning（学习如何做实验）

### 三个核心任务

1. **筛选交互对**：从15个可能的交互对中，找出"有希望"的5-7个
2. **确定超参数**：为后续20个被试的采集函数找到最优配置
3. **验证机制**：检查动态权重（λ_t、γ_t）是否按预期工作

---

## 📊 预热配置

### 基本参数

```python
预热阶段 = {
    "被试数": 8,
    "每人采样": 20次,
    "总样本": 160,
    "采样方法": "Maximin Latin Hypercube Sampling",
    "是否主动学习": False  # 纯Space-filling
}
```

### 为什么是8人×20次？

**被试数选择（8人）：**
```
太少（<5人）：
  - 统计功效不足（交互对筛选不可靠）
  - 个体差异未充分体现
  - 超参数估计可能偏离

合适（8-10人）：
  - 每个交互对约10.7个样本（160÷15=10.7）
  - 混合模型可以给出初步估计
  - 超参数中位数较鲁棒

太多（>12人）：
  - 挤占主动学习预算（机会成本高）
  - 预热目的是"筛选"不是"精确估计"
```

**每人采样数（20次）：**
```
太少（<15次）：
  - GP模型难以学习6维空间
  - r_t下降不明显（无法验证动态机制）

合适（20次）：
  - 足够覆盖主效应的主要水平
  - LHS可以生成良好的Space-filling分布

太多（>25次）：
  - 浪费预算（主动学习更高效）
  - 预热只需"初步覆盖"
```

---

## 🔧 策略A：Maximin LHS（推荐）

### 设计原理

**Latin Hypercube Sampling (LHS)**：
- 将每个维度分成n层
- 每层恰好采样一次
- 保证边际分布均匀

**Maximin优化**：
- 最大化样本间的最小距离
- 避免样本聚集
- 提高空间填充质量

### 生成代码

```python
from scipy.stats import qmc
import numpy as np

def generate_warmup_design(n_factors=6, n_samples=20, seed=42):
    """
    生成Maximin LHS预热设计

    Args:
        n_factors: 因子数量
        n_samples: 每个被试的样本数
        seed: 随机种子（确保可重复）

    Returns:
        X_warmup: (n_samples, n_factors) 的标准化样本 [0,1]
    """
    # 创建LHS采样器
    sampler = qmc.LatinHypercube(d=n_factors, optimization="random-cd", seed=seed)

    # 生成标准化样本 [0, 1]
    X_warmup = sampler.random(n=n_samples)

    # 计算覆盖质量
    from scipy.spatial.distance import pdist
    min_dist = pdist(X_warmup).min()
    print(f"✓ 最小成对距离: {min_dist:.4f}")

    return X_warmup

# 为每个被试生成独立的LHS设计
for subject_id in range(8):
    X_warmup = generate_warmup_design(n_factors=6, n_samples=20, seed=42+subject_id)

    # 映射到实际因子范围
    # X_scaled = map_to_factor_ranges(X_warmup, factor_configs)
```

### 映射到实际因子范围

```python
def map_to_factor_ranges(X_normalized, factor_configs):
    """
    将标准化样本映射到实际因子范围

    Args:
        X_normalized: (n, d) 标准化样本 [0,1]
        factor_configs: 因子配置列表
            例如：[
                {"type": "continuous", "range": [0, 10]},
                {"type": "categorical", "levels": [0, 1, 2]},
                {"type": "integer", "range": [1, 20]}
            ]

    Returns:
        X_mapped: (n, d) 映射后的样本
    """
    n, d = X_normalized.shape
    X_mapped = np.zeros((n, d))

    for i, config in enumerate(factor_configs):
        if config["type"] == "continuous":
            # 连续变量：线性映射
            low, high = config["range"]
            X_mapped[:, i] = X_normalized[:, i] * (high - low) + low

        elif config["type"] == "categorical":
            # 分类变量：分层采样
            levels = config["levels"]
            n_levels = len(levels)
            indices = (X_normalized[:, i] * n_levels).astype(int)
            indices = np.clip(indices, 0, n_levels - 1)
            X_mapped[:, i] = [levels[idx] for idx in indices]

        elif config["type"] == "integer":
            # 整数变量：舍入
            low, high = config["range"]
            X_mapped[:, i] = np.round(X_normalized[:, i] * (high - low) + low)

    return X_mapped
```

---

## 🧪 预热阶段的数据收集

### 流程图

```
对于每个被试 s ∈ {0, 1, ..., 7}:
    1. 初始化独立GP模型 model_s
    2. 生成LHS设计 X_warmup_s (20个样本)
    3. 对于每个样本 x ∈ X_warmup_s:
        a. 运行实验 y = run_trial(subject_id=s, x=x)
        b. 更新模型 model_s.update(x, y)
        c. 记录数据 (x, y, s, trial_number)
    4. 保存被试数据
```

### 数据记录格式

```python
warmup_data = []

for subject_id in range(8):
    model_individual = GPModel()
    X_warmup = generate_warmup_design(n_factors=6, n_samples=20, seed=42+subject_id)
    X_scaled = map_to_factor_ranges(X_warmup, factor_configs)

    for trial, x in enumerate(X_scaled):
        # 运行实验
        y = run_trial(subject_id, x)

        # 更新模型
        model_individual.update(x, y)

        # 记录数据
        warmup_data.append({
            "subject_id": subject_id,
            "trial": trial,
            "phase": "warmup",
            "x0": x[0],
            "x1": x[1],
            "x2": x[2],
            "x3": x[3],
            "x4": x[4],
            "x5": x[5],
            "y": y,
            "timestamp": datetime.now()
        })

# 转换为DataFrame
import pandas as pd
df_warmup = pd.DataFrame(warmup_data)
df_warmup.to_csv("warmup_data.csv", index=False)
```

---

## 📈 Meta-learning分析

### 步骤1：拟合初步混合模型

```python
import statsmodels.formula.api as smf

# 构造包含所有交互的公式
formula = "y ~ x0 + x1 + x2 + x3 + x4 + x5"  # 主效应

# 添加所有15个二阶交互
for i in range(6):
    for j in range(i+1, 6):
        formula += f" + x{i}:x{j}"

# 添加随机效应（随机截距）
formula += " + (1 | subject_id)"

# 拟合模型
preliminary_model = smf.mixedlm(formula, data=df_warmup, groups=df_warmup["subject_id"])
fitted_model = preliminary_model.fit(method="lbfgs")

print(fitted_model.summary())
```

### 步骤2：筛选交互对

```python
# 提取交互项的p值
interaction_results = []

for i in range(6):
    for j in range(i+1, 6):
        param_name = f"x{i}:x{j}"
        if param_name in fitted_model.pvalues.index:
            p_value = fitted_model.pvalues[param_name]
            coef = fitted_model.fe_params[param_name]
            interaction_results.append({
                "pair": (i, j),
                "coef": coef,
                "p_value": p_value,
                "abs_coef": abs(coef)
            })

# 排序并筛选
df_interactions = pd.DataFrame(interaction_results)
df_interactions = df_interactions.sort_values("abs_coef", ascending=False)

# 策略1：使用p值阈值（宽松）
significant_pairs_p = [
    row["pair"] for _, row in df_interactions.iterrows()
    if row["p_value"] < 0.10  # 宽松阈值
]

# 策略2：使用Top-K（更保守）
significant_pairs_topk = [
    row["pair"] for _, row in df_interactions.head(5).iterrows()
]

# 推荐：结合两种策略
# 如果p<0.10的交互对<3个，使用Top-5
# 如果p<0.10的交互对>8个，只保留Top-8
if len(significant_pairs_p) < 3:
    selected_pairs = significant_pairs_topk[:5]
elif len(significant_pairs_p) > 8:
    selected_pairs = [
        row["pair"] for _, row in df_interactions.iterrows()
        if row["p_value"] < 0.10
    ][:8]
else:
    selected_pairs = significant_pairs_p

print(f"✓ 选择的交互对: {selected_pairs}")
```

### 步骤3：确定采集函数超参数

```python
# 3.1 分析每个被试的r_t轨迹（如果记录了）
# 这需要在预热过程中跟踪参数方差
# 略过（需要修改数据收集代码）

# 3.2 确定lambda_max
# 方法：基于交互效应的相对重要性
main_effects_var = fitted_model.fe_params[["x0", "x1", "x2", "x3", "x4", "x5"]].var()
interaction_effects_var = df_interactions["abs_coef"].var()

if interaction_effects_var > main_effects_var * 0.5:
    recommended_lambda_max = 1.0  # 交互很重要
elif interaction_effects_var > main_effects_var * 0.2:
    recommended_lambda_max = 0.8  # 交互中等重要
else:
    recommended_lambda_max = 0.5  # 交互不太重要

print(f"✓ 推荐lambda_max: {recommended_lambda_max}")

# 3.3 确定tau_n_max
# 规则：主动学习预算的70%位置
n_active_trials = 30  # 每个被试30次采样
recommended_tau_n_max = int(n_active_trials * 0.7)
print(f"✓ 推荐tau_n_max: {recommended_tau_n_max}")

# 3.4 确定gamma_min
# 规则：小样本预算使用较小的gamma_min
if n_active_trials < 30:
    recommended_gamma_min = 0.05
else:
    recommended_gamma_min = 0.1
print(f"✓ 推荐gamma_min: {recommended_gamma_min}")
```

---

## ✅ 预热阶段检查清单

### 数据收集前

- [ ] 确认因子类型和范围（连续/分类/整数）
- [ ] 生成并验证LHS设计（检查最小距离）
- [ ] 测试GP模型配置（核函数、似然函数）
- [ ] 准备数据记录代码（CSV或数据库）

### 数据收集中

- [ ] 每个被试独立建模（不共享数据）
- [ ] 实时更新GP模型（每次采样后update）
- [ ] 记录完整元数据（被试ID、轮次、时间戳）
- [ ] 备份数据（防止丢失）

### 数据收集后

- [ ] 数据完整性检查（160个样本，8个被试）
- [ ] 拟合初步混合模型（检查收敛性）
- [ ] 筛选交互对（使用p<0.10或Top-K）
- [ ] 确定超参数（lambda_max, tau_n_max, gamma_min）
- [ ] 生成Meta-learning报告

---

## 📊 预期结果

### 正常情况

```
✓ 8个被试完成数据收集（160样本）
✓ 混合模型收敛（AIC/BIC合理）
✓ 筛选出5-7个交互对（p<0.10）
✓ 主效应系数显著（至少4个p<0.05）
✓ 随机效应方差合理（不为0）
```

### 异常情况处理

**情况1：没有显著交互（所有p>0.10）**
```
原因：
  - 真实情况下可能确实无交互
  - 样本量不足（统计功效低）

处理：
  - 使用Top-5交互对（基于系数大小）
  - 在主动学习中仍配置这些交互对
  - 最终分析会给出准确的p值
```

**情况2：模型不收敛**
```
原因：
  - GP模型配置问题（核函数、超参数）
  - 数据质量问题（异常值、缺失值）

处理：
  - 检查数据（可视化、异常值检测）
  - 调整GP超参数（lengthscale、noise）
  - 简化混合模型（只用随机截距）
```

**情况3：r_t未下降**
```
原因：
  - LHS覆盖不足（样本太少）
  - 模型未充分学习

处理：
  - 增加预热样本数（20→25）
  - 检查GP训练过程（是否正确update）
```

---

## 🔬 预热质量评估指标

### 覆盖度指标

```python
# 计算最小成对距离（应>0.1）
from scipy.spatial.distance import pdist
min_dist = pdist(X_warmup).min()

# 计算空间覆盖率（基于Voronoi图）
from scipy.spatial import Voronoi
vor = Voronoi(X_warmup)
# （具体计算略）
```

### 模型质量指标

```python
# 混合模型R²
R2_marginal = fitted_model.rsquared_marginal  # 固定效应R²
R2_conditional = fitted_model.rsquared_conditional  # 总R²

# 应满足：
# - R2_marginal > 0.3（固定效应有解释力）
# - R2_conditional > 0.5（总体拟合良好）
```

### 交互对可靠性

```python
# 一致性检查：Bootstrap重采样
from sklearn.utils import resample

bootstrap_results = []
for _ in range(100):
    # 重采样被试
    resampled_subjects = resample(range(8), n_samples=8)
    df_resampled = df_warmup[df_warmup["subject_id"].isin(resampled_subjects)]

    # 重新拟合
    model_boot = smf.mixedlm(formula, data=df_resampled, groups=df_resampled["subject_id"]).fit()

    # 记录显著交互对
    sig_pairs = [
        (i, j) for i in range(6) for j in range(i+1, 6)
        if model_boot.pvalues.get(f"x{i}:x{j}", 1.0) < 0.10
    ]
    bootstrap_results.append(sig_pairs)

# 计算每个交互对的选中率
from collections import Counter
all_pairs = [pair for result in bootstrap_results for pair in result]
pair_stability = Counter(all_pairs)

# 推荐：只选择选中率>60%的交互对
stable_pairs = [pair for pair, count in pair_stability.items() if count/100 > 0.6]
```

---

## 📝 Meta-learning报告模板

```markdown
# 预热阶段Meta-learning报告

## 基本信息
- 完成日期：YYYY-MM-DD
- 被试数：8
- 总样本：160
- 数据文件：warmup_data.csv

## 混合模型结果
- AIC: XXXX
- BIC: XXXX
- R² (marginal): 0.XX
- R² (conditional): 0.XX

## 主效应估计
| 因子 | 系数 | p值 | 显著性 |
|------|------|-----|--------|
| x0 | 0.XX | 0.XXX | ** |
| x1 | 0.XX | 0.XXX | * |
| ... | ... | ... | ... |

## 交互效应筛选
| 交互对 | 系数 | p值 | 选中 |
|--------|------|-----|------|
| (0,1) | 0.XX | 0.XXX | ✓ |
| (2,5) | 0.XX | 0.XXX | ✓ |
| ... | ... | ... | ... |

**选中的交互对（5个）**：(0,1), (2,5), (0,3), (1,4), (3,5)

## 推荐超参数
- lambda_max: 0.8
- tau_n_max: 21
- gamma_min: 0.05
- tau1: 0.7（默认）
- tau2: 0.3（默认）

## 质量指标
- 最小成对距离: 0.15 ✓
- Bootstrap稳定性: 平均选中率 0.68 ✓
- 模型收敛: 是 ✓

## 建议
进入主动学习阶段，使用以上配置。
```

---

## 🔗 相关文档

- **核心思路**：`00_CORE_IDEAS.md`
- **完整实验设计**：`../EXPERIMENT_DESIGN.md`
- **预热代码示例**：`../warmup_strategy_example.py`
- **策略对比**：`../strategy_matrix_critique.md`

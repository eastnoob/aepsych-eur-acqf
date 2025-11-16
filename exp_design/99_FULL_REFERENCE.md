# 自适应采样实验设计方案

## 📋 实验背景

- **研究类型**：探索性心理学实验
- **总被试数**：30人
- **每人采样**：20-30次
- **因子数量**：6个混合变量（分类/整数/连续）
- **核心挑战**：
  - ✓ 有限采样预算（总样本 ~750-900）
  - ✓ 存在个体差异（每人的心理函数可能不同）
  - ✓ 交互效应未知（C(6,2)=15个可能交互对）

---

## 🎯 核心设计思路

### 两阶段分离原则

```
【采样阶段】GP + 自适应采集函数
   目的：在有限预算下，找到信息量最大的采样点
   工具：EURAnovaPairAcqf（动态权重机制）

【分析阶段】混合效应线性模型
   目的：估计效应量、检验显著性、处理个体差异
   工具：statsmodels.mixedlm 或 R lme4
```

**关键认知**：
- GP擅长"在哪里采样"（探索-利用平衡）
- 线性模型擅长"效应有多大"（统计推断）
- 每个被试独立建模（避免过拟合，尊重个体差异）

---

## 📐 三阶段实验计划

### 阶段1：预热与Meta-learning（8人 × 20次 = 160样本）

**目标**：
1. 筛选"有希望"的交互对（避免浪费样本在不存在的交互上）
2. 确定采集函数超参数（λ_max, γ, tau_n_max）
3. 为主动学习阶段提供先验信息

**采样方案**：
```python
预热设计 = {
    "方法": "Maximin Latin Hypercube Sampling（策略A）",
    "样本数": 20次/人 × 8人 = 160样本,
    "特点": "纯Space-filling（不使用主动学习）",
    "原因": "需要无偏的初步估计，避免采样偏差"
}
```

**数据分析**：
```python
# 拟合初步混合效应模型
preliminary_model = mixedlm(
    formula="y ~ x0 + x1 + ... + x5 + x0:x1 + ... + x4:x5 + (1|subject_id)",
    data=warmup_data
).fit()

# Meta-learning步骤：
1. 筛选交互对：选择 p < 0.10 的交互对（宽松阈值）
   → 预期得到 5-7 个"候选交互对"

2. 确定超参数：
   - lambda_max = median(各被试的最优值)
   - tau_n_max = 主动学习预算的70%位置（例如30次×0.7=21）
   - gamma_min = 0.05（小样本预算）

3. 检查r_t轨迹：验证动态权重机制是否按预期工作
```

**为什么选择策略A（纯Space-filling）？**
- ✅ 均匀覆盖所有因子水平（主效应估计准确）
- ✅ 为交互检测提供无偏基线
- ✅ 与动态λ_t机制协同（预热后r_t↓ → 自动提高交互权重）
- ✅ 避免预先假设哪些交互存在（数据驱动）

---

### 阶段2：主动学习（20人 × 30次 = 600样本）

**目标**：
1. 高效探索"预热阶段筛选出的交互对"
2. 动态平衡探索与精细化（λ_t和γ_t自适应）
3. 每个被试独立建模（尊重个体差异）

**采样方案**：
```python
每个被试的采样配置 = {
    "前5次": "纯随机采样（避免主动学习偏差）",
    "后25次": "主动学习（GP + EURAnovaPairAcqf）"
}

采集函数配置 = {
    "interaction_pairs": [预热阶段筛选出的5-7个交互对],
    "lambda_max": [从预热中学到的值],
    "tau_n_max": 21,  # 在70%位置转向精细化
    "gamma_min": 0.05,
    "use_dynamic_lambda": True,  # 启用交互权重自适应
    "use_dynamic_gamma": True    # 启用信息/覆盖自适应
}
```

**关键机制**：
```
每个被试的数据流：
  样本1-5（随机）→ GP学习基础趋势 → r_t逐渐下降
  样本6-21（主动）→ λ_t逐渐提高 → 探索交互区域
  样本22-30（主动）→ γ_t降低 → 聚焦精细化估计
```

**为什么需要"前5次随机"？**
- 主动学习的样本不是随机的（会集中在高不确定性区域）
- 混合模型需要一定的随机样本保证估计无偏
- 5次占比17%（可接受的效率损失）

---

### 阶段3：验证（可选，2人 × 30次 = 60样本）

**目标**：
检验主动学习是否引入显著偏差

**采样方案**：
```python
验证设计 = {
    "方法": "纯随机采样",
    "样本数": 30次/人 × 2人 = 60样本,
    "目的": "提供无偏的对照组"
}
```

**偏差检验**：
```python
# 拟合两个混合模型
model_main = fit_mixed_effects(预热+主动学习数据)    # 660样本
model_validation = fit_mixed_effects(验证数据)        # 60样本

# 比较固定效应估计
compare_estimates(
    model_main.fe_params,
    model_validation.fe_params,
    tolerance=0.2  # 允许20%差异
)

# 如果差异>20% → 考虑增加随机样本比例（例如前10次随机）
```

**是否必须？**
- 预算充足：推荐（增加可信度）
- 预算紧张：可跳过（但需在论文中说明局限性）

---

## 📊 最终数据分析

### 混合效应模型配置

```python
import statsmodels.formula.api as smf
import pandas as pd

# 整合所有数据
all_data = pd.concat([
    warmup_data,      # 160样本（8人×20次）
    active_data,      # 600样本（20人×30次）
    validation_data   # 60样本（2人×30次，可选）
])

# 固定效应：主效应 + 预热中显著的交互
fixed_effects = "x0 + x1 + x2 + x3 + x4 + x5"
for i, j in significant_pairs:  # 5-7个交互对
    fixed_effects += f" + x{i}:x{j}"

# 随机效应：随机截距 + 随机斜率（考虑个体差异）
random_effects = "(1 + x0 + x1 + x2 + x3 + x4 + x5 | subject_id)"

# 拟合模型
formula = f"y ~ {fixed_effects} + {random_effects}"
final_model = smf.mixedlm(
    formula,
    data=all_data,
    groups=all_data["subject_id"]
).fit(method="lbfgs")

# 报告结果
print(final_model.summary())
print(f"\n固定效应（群体水平）：")
print(final_model.fe_params)  # 系数
print(final_model.pvalues)    # p值
print(final_model.conf_int()) # 95%置信区间

print(f"\n随机效应方差（个体差异）：")
print(final_model.cov_re)
```

### 报告内容（APA格式）

1. **主效应**：
   - 系数估计 ± 标准误
   - p值与显著性标记（*p<.05, **p<.01, ***p<.001）
   - 效应量（Cohen's d 或 η²）

2. **交互效应**：
   - 显著交互对的系数
   - 简单斜率分析（simple slope analysis）
   - 交互图（interaction plot）

3. **个体差异**：
   - 随机效应方差估计
   - 组内相关系数（ICC）
   - 个体轨迹图（spaghetti plot）

4. **模型拟合**：
   - AIC/BIC（模型比较）
   - R²（边际和条件）
   - 残差诊断

---

## 📈 样本预算分配总结

| 阶段 | 被试数 | 每人采样 | 总样本 | 占比 | 用途 |
|------|-------|---------|-------|------|------|
| **预热** | 8 | 20（LHS） | 160 | 20% | Meta-learning（筛选交互对） |
| **主动** | 20 | 5随机+25主动 | 600 | 75% | 高效探索交互区域 |
| **验证** | 2 | 30（随机） | 60 | 5% | 偏差检验（可选） |
| **合计** | 30 | - | 820 | 100% | 最终混合模型分析 |

---

## 🔧 关键参数配置参考

### 采集函数参数（从预热中学习）

```python
# 示例配置（实际值从预热数据中确定）
acqf_config = {
    # 交互对（从预热中筛选）
    "interaction_pairs": [(0,1), (2,5), (0,3), (1,4), (3,5)],  # 5个交互对

    # 动态权重参数（λ_t）
    "use_dynamic_lambda": True,
    "lambda_min": 0.1,   # 参数未收敛时的交互权重
    "lambda_max": 1.0,   # 参数已收敛时的交互权重
    "tau1": 0.7,         # r_t上阈值
    "tau2": 0.3,         # r_t下阈值

    # 动态覆盖参数（γ_t）
    "use_dynamic_gamma": True,
    "gamma_max": 0.5,    # 初期重视覆盖
    "gamma_min": 0.05,   # 后期重视信息
    "tau_n_min": 3,      # 最小样本数阈值
    "tau_n_max": 21,     # 70%位置（30次×0.7）

    # 局部扰动
    "local_jitter_frac": 0.1,
    "local_num": 4,

    # 调试
    "debug_components": True  # 监控λ_t和γ_t的动态变化
}
```

### 预热阶段的LHS参数

```python
from scipy.stats import qmc

# Maximin LHS生成
lhs_sampler = qmc.LatinHypercube(d=6, optimization="maximin", seed=42)
lhs_samples = lhs_sampler.random(n=20)  # 每个被试20个样本

# 映射到实际因子范围
factor_ranges = [
    (0, 1),  # 因子0范围
    (0, 1),  # 因子1范围
    # ...
]
X_warmup = qmc.scale(lhs_samples, l_bounds, u_bounds)
```

---

## ⚠️ 注意事项

### 1. 数据收集过程
- ✅ 每个被试使用独立的GP模型（避免过拟合）
- ✅ 实时更新模型（每次采样后立即update）
- ✅ 记录完整的元数据（被试ID、采样轮次、阶段标签）

### 2. 采集函数监控
```python
# 每个被试结束后检查诊断信息
if trial % 30 == 0:
    diag = acqf.get_diagnostics()
    print(f"被试 {subject_id}:")
    print(f"  r_t = {diag['r_t']:.3f}")      # 应从1.0逐渐下降到<0.3
    print(f"  λ_t = {diag['lambda_t']:.3f}")  # 应逐渐提高到1.0
    print(f"  γ_t = {diag['gamma_t']:.3f}")   # 应从0.5逐渐降到0.05
```

### 3. 常见问题
- **Q**: 如果预热后没有显著交互怎么办？
  - **A**: 在主动学习中使用所有15个交互对（保守策略）

- **Q**: 如果某个被试的r_t一直很高怎么办？
  - **A**: 检查模型配置（核函数、超参数优化），可能需要增加随机样本

- **Q**: 验证阶段发现显著偏差怎么办？
  - **A**: 增加每个被试的随机样本比例（5次→10次）

---

## 📚 理论支撑

1. **Cavagnaro et al. (2013)** - Adaptive Design Optimization in Experiments with People
   → GP主动采样 + 传统统计推断的结合

2. **Owen et al. (2021)** - AEPsych: An Open-Source Software Package
   → 心理物理实验中的自适应采样框架

3. **Bates et al. (2015)** - Fitting Linear Mixed-Effects Models Using lme4
   → 混合效应模型处理个体差异的标准方法

4. **Box & Draper (1987)** - Empirical Model-Building and Response Surfaces
   → Space-filling设计在探索性研究中的优势

---

## ✅ 检查清单

实验开始前：
- [ ] 确认6个因子的类型（分类/整数/连续）和范围
- [ ] 生成预热阶段的LHS样本（8人×20次）
- [ ] 配置GP模型（核函数、似然函数）
- [ ] 测试采集函数（小样本验证代码正确性）

预热阶段后：
- [ ] 拟合初步混合模型（检查收敛性）
- [ ] 筛选显著交互对（p<0.10阈值）
- [ ] 确定采集函数超参数（λ_max, tau_n_max等）
- [ ] 记录r_t轨迹（验证动态机制）

主动学习阶段：
- [ ] 每个被试独立建模
- [ ] 前5次使用随机采样
- [ ] 监控λ_t和γ_t的变化
- [ ] 记录采样点和响应（含时间戳）

数据分析阶段：
- [ ] 整合所有数据（含阶段标签）
- [ ] 拟合最终混合模型（检查残差）
- [ ] 报告APA格式的结果（系数、p值、置信区间）
- [ ] 可视化（交互图、个体轨迹图）

---

## 📧 问题反馈

如有疑问，请参考：
- 采集函数文档：`eur_anova_pair.py` 开头的docstring
- 策略对比分析：`strategy_matrix_critique.md`
- 预热策略示例：`warmup_strategy_example.py`

# 预热阶段策略：深度分析与实施方案

## 🎯 核心洞察：你的采集函数告诉我们什么

通过分析你的 `EURAnovaPairAcqf` 代码，我发现了关键机制：

### 1. 参数方差率 r_t 是核心控制变量
```python
r_t = Var[θ|D_current] / Var[θ|D_initial]  # 第952行
```
- **r_t ≈ 1.0**：参数未收敛，λ_t → λ_min (0.1)，**主效应优先**
- **r_t < 0.3**：参数已收敛，λ_t → λ_max (1.0)，**交互效应优先**

### 2. 动态权重机制的含义
你的采集函数通过 λ_t 自动调整主效应与交互效应的探索权重：
- **早期（r_t高）**：专注主效应，建立稳定基线
- **后期（r_t低）**：转向交互效应，精细化探索

**关键启示**：预热阶段应该让 r_t 快速下降，这样主动学习阶段才能有效探索交互！

---

## 📊 预热策略推荐：**策略A - 纯Space-filling**

### 为什么选择Space-filling而非其他策略？

| 策略 | 主效应估计 | r_t下降速度 | 交互筛选质量 | 推荐度 |
|------|-----------|------------|-------------|---------|
| **A: Space-filling (LHS)** | ★★★★★ | ★★★★★ | ★★★★ | **最优** |
| B: D-optimal | ★★★★ | ★★★ | ★★ | 次选 |
| C: 随机采样 | ★★ | ★★ | ★★ | 不推荐 |

### 选择LHS的理论依据：

1. **均匀覆盖 → 主效应准确估计**
   - LHS保证每个因子的每个水平都被充分采样
   - 主效应方差快速下降 → r_t快速下降

2. **r_t快速下降 → λ_t自动上升**
   - 无需手动调整，你的动态机制会自动工作
   - 预热结束时，λ_t会自然升高，为交互探索做好准备

3. **无偏交互筛选**
   - Space-filling避免了采样偏差
   - 交互效应的p值更可靠

---

## 🔧 预热阶段实施方案

### Step 1: 生成LHS采样点（160个）

```python
from pyDOE2 import lhs
from scipy.spatial.distance import pdist
import numpy as np

def generate_maximin_lhs(n_samples=160, n_factors=6, n_iterations=100):
    """生成Maximin LHS设计"""
    best_design = None
    best_min_dist = -1
    
    for _ in range(n_iterations):
        # 生成LHS
        design = lhs(n_factors, samples=n_samples, criterion='maximin')
        
        # 计算最小距离
        min_dist = pdist(design).min()
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_design = design
    
    return best_design

# 生成预热设计
X_warmup = generate_maximin_lhs(160, 6)
```

### Step 2: 收集数据（8人×20次）

```python
# 分配给8个被试
subject_assignments = np.repeat(np.arange(8), 20)
np.random.shuffle(subject_assignments)  # 随机分配

for i, subject_id in enumerate(subject_assignments):
    x_trial = X_warmup[i]
    y_response = collect_response(subject_id, x_trial)
    save_data(subject_id, x_trial, y_response)
```

### Step 3: 分析预热数据 - **关键步骤**

```python
import statsmodels.formula.api as smf
import pandas as pd

def analyze_warmup_data(X_warmup, y_warmup, subject_ids):
    """
    分析预热数据，提取关键信息用于主动学习配置
    
    Returns:
        dict: 包含筛选的交互对、参数估计、超参数建议
    """
    
    # 1. 构建数据框
    df = pd.DataFrame(X_warmup, columns=[f'x{i}' for i in range(6)])
    df['y'] = y_warmup
    df['subject'] = subject_ids
    
    # 2. 拟合混合效应模型（主效应 + 所有二阶交互）
    formula = 'y ~ x0 + x1 + x2 + x3 + x4 + x5'  # 主效应
    
    # 添加所有可能的交互项
    interactions = []
    for i in range(6):
        for j in range(i+1, 6):
            interactions.append(f'x{i}:x{j}')
    
    formula += ' + ' + ' + '.join(interactions)
    formula += ' + (1|subject)'  # 随机截距
    
    model = smf.mixedlm(formula, df, groups=df["subject"])
    result = model.fit()
    
    # 3. 筛选显著交互（p < 0.10）
    significant_pairs = []
    for i in range(6):
        for j in range(i+1, 6):
            coef_name = f'x{i}:x{j}'
            if coef_name in result.params:
                p_value = result.pvalues[coef_name]
                if p_value < 0.10:  # 宽松阈值
                    significant_pairs.append((i, j))
                    print(f"交互 ({i},{j}): β={result.params[coef_name]:.3f}, p={p_value:.3f}")
    
    # 4. 计算主效应方差（用于估计r_t）
    main_effects_var = {}
    for i in range(6):
        coef_name = f'x{i}'
        if coef_name in result.params:
            # 使用标准误的平方作为方差估计
            main_effects_var[i] = result.bse[coef_name] ** 2
    
    # 5. 估计初始r_t（为主动学习提供基线）
    avg_var = np.mean(list(main_effects_var.values()))
    
    # 6. 超参数建议
    hyperparams = {
        'lambda_max': 1.0 if len(significant_pairs) > 3 else 0.7,  # 交互多则权重高
        'tau_n_max': 21,  # 70%位置转向精细化
        'tau1': 0.7,  # r_t上阈值
        'tau2': 0.3,  # r_t下阈值
        'initial_variance_estimate': avg_var
    }
    
    return {
        'interaction_pairs': significant_pairs,
        'main_effects': {f'x{i}': result.params[f'x{i}'] for i in range(6)},
        'main_effects_se': {f'x{i}': result.bse[f'x{i}'] for i in range(6)},
        'hyperparams': hyperparams,
        'model_summary': result.summary()
    }
```

### Step 4: 配置主动学习阶段

```python
# 使用预热分析结果配置采集函数
warmup_results = analyze_warmup_data(X_warmup, y_warmup, subject_ids)

# 为每个后续被试配置采集函数
def create_acquisition_function(model, warmup_results):
    """基于预热结果创建优化的采集函数"""
    
    return EURAnovaPairAcqf(
        model=model,
        # 使用筛选的交互对
        interaction_pairs=warmup_results['interaction_pairs'],
        
        # 使用优化的超参数
        lambda_max=warmup_results['hyperparams']['lambda_max'],
        lambda_min=0.1,
        tau1=warmup_results['hyperparams']['tau1'],
        tau2=warmup_results['hyperparams']['tau2'],
        tau_n_max=warmup_results['hyperparams']['tau_n_max'],
        
        # 动态权重机制
        use_dynamic_lambda=True,
        use_dynamic_gamma=True,
        
        # 其他参数
        gamma=0.3,
        gamma_min=0.05,
        total_budget=30,  # 每个被试30次
    )
```

---

## 📈 预热产出清单

### 必须获得的统计量：

1. **交互对筛选列表**
   - 5-7个 p<0.10 的交互对
   - 格式：`[(0,1), (2,5), (3,4), ...]`

2. **主效应估计**
   - 每个因子的系数和标准误
   - 用于验证模型收敛性

3. **参数方差基线**
   - 初始方差估计，用于r_t计算
   - 判断何时从探索转向精细化

4. **超参数配置**
   - `lambda_max`：基于交互对数量
   - `tau_n_max`：转向点设置

### 可选但有价值的分析：

1. **个体差异评估**
   ```python
   # 随机效应方差
   random_var = result.cov_re
   ICC = random_var / (random_var + residual_var)
   print(f"组内相关系数ICC: {ICC:.3f}")
   ```

2. **效应量排序**
   ```python
   # 标准化效应量
   effect_sizes = {}
   for param, value in result.params.items():
       if ':' not in param and param != 'Intercept':
           effect_sizes[param] = abs(value) / result.bse[param]
   
   sorted_effects = sorted(effect_sizes.items(), key=lambda x: x[1], reverse=True)
   ```

3. **设计诊断**
   ```python
   # VIF检查（多重共线性）
   from statsmodels.stats.outliers_influence import variance_inflation_factor
   
   vif = [variance_inflation_factor(X_warmup, i) for i in range(6)]
   print(f"VIF值: {vif}")  # 应该都<5
   ```

---

## ⚠️ 关键注意事项

### 1. 不要过度筛选交互
- 使用 p<0.10 而非 p<0.05
- 宁可包含假阳性，不要错过真实效应
- 你的动态λ_t会自动降低不重要交互的权重

### 2. 保持预热的纯粹性
- **不要**在预热阶段使用主动学习
- **不要**根据中期结果调整设计
- 坚持完成全部160个LHS点

### 3. 正确使用预热结果
- 预热是为了**配置**主动学习，不是为了**得出结论**
- 效应估计是初步的，最终分析使用全部数据

---

## 💡 最终建议

你的采集函数设计精妙，动态机制会自动适应数据。预热阶段的核心任务是：

1. **让r_t快速下降**（通过LHS均匀采样）
2. **筛选候选交互对**（避免浪费探索）
3. **提供超参数基线**（优化后续表现）

记住：**LHS预热 → r_t快速下降 → λ_t自动上升 → 交互探索自然开启**

这是一个自洽的系统，预热做好了，后面的主动学习会自动高效运行！

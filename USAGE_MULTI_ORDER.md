# 多阶交互ANOVA采集函数使用指南

## 🎯 快速开始

### 基础示例：只启用主效应

```python
from dynamic_eur_acquisition import EURAnovaMultiAcqf

# 创建采集函数（只探索主效应）
acqf = EURAnovaMultiAcqf(
    model,
    enable_main=True,
    enable_pairwise=False,  # 关闭二阶交互
    enable_threeway=False   # 关闭三阶交互
)

# 评估候选点
scores = acqf(X_candidates)
```

### 启用二阶交互

```python
acqf = EURAnovaMultiAcqf(
    model,
    enable_main=True,
    interaction_pairs=[(0, 1), (2, 3)],  # 指定二阶交互对
    lambda_2=1.0  # 二阶权重（默认动态调整）
)
```

### 完整配置：主效应 + 二阶 + 三阶

```python
acqf = EURAnovaMultiAcqf(
    model,
    # 启用配置
    enable_main=True,
    enable_pairwise=True,
    enable_threeway=True,

    # 交互配置
    interaction_pairs=[(0, 1), (1, 2), (2, 3)],
    interaction_triplets=[(0, 1, 2)],

    # 权重配置
    main_weight=1.0,
    lambda_2=1.0,    # 二阶权重（None=动态）
    lambda_3=0.5,    # 三阶权重（推荐较小值）

    # 动态权重（如果lambda_2=None）
    use_dynamic_lambda=True,
    tau1=0.7,        # 参数方差上阈值
    tau2=0.3,        # 参数方差下阈值
    lambda_min=0.1,
    lambda_max=1.0,

    # 覆盖度权重
    gamma=0.3,
    use_dynamic_gamma=True,
    gamma_min=0.05,
    gamma_max=0.5,

    # 混合变量类型
    variable_types={
        0: 'categorical',
        1: 'continuous',
        2: 'integer',
        3: 'continuous'
    },

    # 调试
    debug_components=True
)

# 评估
scores = acqf(X_candidates)

# 查看诊断信息
acqf.print_diagnostics()
```

---

## 📊 配置策略

### 1. 探索性研究（推荐默认）

**场景**：不知道哪些交互存在，需要全面探索

```python
acqf = EURAnovaMultiAcqf(
    model,
    enable_main=True,
    interaction_pairs='all',  # 或手动指定所有可能的对
    enable_threeway=False,    # 预算有限时关闭三阶
    lambda_2=None,            # 动态调整
    debug_components=True
)
```

### 2. 验证性研究（已知交互）

**场景**：从先导实验得知特定交互，需要精确估计

```python
acqf = EURAnovaMultiAcqf(
    model,
    enable_main=True,
    interaction_pairs=[(0, 1), (2, 3)],  # 只关注已知交互
    lambda_2=1.0,                        # 固定权重
    use_dynamic_lambda=False
)
```

### 3. 有限预算（<30次采样）

**场景**：样本非常有限，优先保证覆盖度

```python
acqf = EURAnovaMultiAcqf(
    model,
    enable_main=True,
    enable_pairwise=False,  # 预算不足时只探索主效应
    total_budget=20,        # 自动配置tau_n_max和gamma_min
    gamma_max=0.7,          # 提高覆盖权重
    debug_components=True
)
```

### 4. 充足预算（>50次采样）

**场景**：可以深入探索高阶交互

```python
acqf = EURAnovaMultiAcqf(
    model,
    enable_main=True,
    interaction_pairs=[(0, 1), (1, 2), (2, 3), (0, 3)],
    interaction_triplets=[(0, 1, 2), (1, 2, 3)],
    lambda_2=1.0,
    lambda_3=0.5,
    total_budget=60
)
```

---

## 🔧 高级功能

### 动态启用/禁用交互阶数

```python
# 初始化时关闭所有交互
acqf = EURAnovaMultiAcqf(
    model,
    enable_main=True,
    enable_pairwise=False,
    enable_threeway=False
)

# 运行时无法动态更改（需重新初始化）
# 但可以通过配置文件在不同实验阶段使用不同策略
```

### 配置文件驱动

```ini
# config.ini
[acquisition]
enable_main = true
enable_pairwise = true
enable_threeway = false

interaction_pairs = 0,1; 2,3
lambda_2 = 1.0
lambda_3 = 0.5

variable_types = categorical, continuous, integer, continuous

[dynamic_weights]
use_dynamic_lambda = true
tau1 = 0.7
tau2 = 0.3
lambda_min = 0.1
lambda_max = 1.0
```

```python
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

acqf = EURAnovaMultiAcqf(
    model,
    enable_main=config.getboolean('acquisition', 'enable_main'),
    enable_pairwise=config.getboolean('acquisition', 'enable_pairwise'),
    enable_threeway=config.getboolean('acquisition', 'enable_threeway'),
    interaction_pairs=config.get('acquisition', 'interaction_pairs'),
    lambda_2=config.getfloat('acquisition', 'lambda_2'),
    variable_types_list=config.get('acquisition', 'variable_types'),
    use_dynamic_lambda=config.getboolean('dynamic_weights', 'use_dynamic_lambda'),
    tau1=config.getfloat('dynamic_weights', 'tau1'),
    tau2=config.getfloat('dynamic_weights', 'tau2')
)
```

### 诊断和调试

```python
# 启用调试模式
acqf = EURAnovaMultiAcqf(model, debug_components=True)

# 评估后查看详细信息
scores = acqf(X_candidates)

# 获取诊断字典
diag = acqf.get_diagnostics()
print(f"二阶权重: {diag['lambda_2']}")
print(f"三阶权重: {diag['lambda_3']}")
print(f"训练样本数: {diag['n_train']}")

# 打印格式化诊断
acqf.print_diagnostics(verbose=True)
```

输出示例：
```
======================================================================
EURAnovaPairAcqf 诊断信息
======================================================================

【动态权重状态】
  λ_2 (二阶交互权重) = 0.8500
  λ_3 (三阶交互权重) = 0.5000
  γ_t (覆盖权重) = 0.3200
  λ 范围: [0.10, 1.00]
  γ 范围: [0.05, 0.50]

【模型状态】
  训练样本数: 25
  转向阈值: tau_n_min=3, tau_n_max=25
  模型已拟合: 是

【交互配置】
  二阶交互数量: 3
  二阶交互: (0,1), (1,2), (2,3)
  三阶交互数量: 1
  三阶交互: (0,1,2)

【效应贡献】(最后一次 forward() 调用)
  主效应总和: mean=0.1234, std=0.0456
  二阶交互总和: mean=0.0567, std=0.0123
  三阶交互总和: mean=0.0123, std=0.0045
  信息项: mean=0.2345, std=0.0678
  覆盖项: mean=0.5678, std=0.1234
======================================================================
```

---

## 🧪 与AEPsych集成

### 基础集成

```python
import aepsych
from dynamic_eur_acquisition import EURAnovaMultiAcqf

# 在AEPsych配置中使用
config_str = """
[common]
stimuli_per_trial = 1
outcome_types = binary

[EURAnovaMultiAcqf]
enable_main = True
enable_pairwise = True
interaction_pairs = 0,1; 2,3
lambda_2 = 1.0
variable_types = continuous, continuous, continuous, continuous

[experiment]
acqf = EURAnovaMultiAcqf
model = GPClassificationModel
"""
```

---

## 📈 性能考虑

### 计算复杂度

- **主效应**: O(d) 次局部扰动
- **二阶交互**: O(|pairs|) 次局部扰动
- **三阶交互**: O(|triplets|) 次局部扰动

**总模型调用次数**: 1次（批量评估）

### 推荐配置

| 维度数 | 预算 | 推荐配置 |
|--------|------|----------|
| 2-4    | <20  | 只主效应 |
| 2-4    | 20-40 | 主 + 部分二阶 |
| 2-4    | >40  | 主 + 全二阶 + 部分三阶 |
| 5-8    | <30  | 只主效应 |
| 5-8    | 30-60 | 主 + 部分二阶 |
| 5-8    | >60  | 主 + 全二阶 |
| >8     | <50  | 只主效应 |
| >8     | 50-100 | 主 + 重点二阶 |

---

## 🔬 模块化使用（高级）

如果你只需要特定模块：

```python
# 单独使用ANOVA引擎
from dynamic_eur_acquisition.modules import ANOVAEffectEngine, MainEffect, PairwiseEffect

engine = ANOVAEffectEngine(metric_fn, local_sampler)
effects = [MainEffect(0), MainEffect(1), PairwiseEffect(0, 1)]
results = engine.compute_effects(X_candidates, effects)

# 单独使用动态权重系统
from dynamic_eur_acquisition.modules import DynamicWeightEngine

weight_engine = DynamicWeightEngine(model)
lambda_t = weight_engine.compute_lambda()
gamma_t = weight_engine.compute_gamma()

# 单独使用序数模型辅助
from dynamic_eur_acquisition.modules import OrdinalMetricsHelper

ordinal_helper = OrdinalMetricsHelper(model)
if ordinal_helper.is_ordinal():
    entropy = ordinal_helper.compute_entropy(X_candidates)
```

---

## 📚 迁移指南

### 从旧版 `EURAnovaPairAcqf` 迁移

```python
# 旧版（仍支持，向后兼容）
from dynamic_eur_acquisition import EURAnovaPairAcqf
acqf_old = EURAnovaPairAcqf(model, interaction_pairs=[(0,1)])

# 新版（推荐）
from dynamic_eur_acquisition import EURAnovaMultiAcqf
acqf_new = EURAnovaMultiAcqf(
    model,
    enable_main=True,
    interaction_pairs=[(0, 1)],
    enable_threeway=False  # 新增：可选三阶
)

# 行为完全一致（如果不启用三阶）
```

---

## ❓ 常见问题

**Q: 何时启用三阶交互？**

A: 仅在以下情况：
- 预算充足（>50次）
- 先导研究表明存在复杂交互
- 维度较少（≤4维）

**Q: lambda_2 和 lambda_3 如何设置？**

A:
- `lambda_2=None`: 动态调整（推荐探索性研究）
- `lambda_2=1.0`: 固定权重（推荐验证性研究）
- `lambda_3=0.5`: 三阶权重建议<1.0（避免过拟合）

**Q: 如何判断是否需要交互项？**

A:
1. 先运行只主效应的实验
2. 查看主效应贡献是否能解释大部分不确定性
3. 如果主效应不足，逐步添加二阶交互
4. 只有在二阶仍不足时才考虑三阶

**Q: 性能如何？**

A: 批量优化后，20维+15个二阶+1个三阶的配置，单次评估仍只需1次模型调用（vs 原始实现的36次）。

---

## 🎉 完整示例：心理物理实验

```python
from dynamic_eur_acquisition import EURAnovaMultiAcqf
from aepsych.models import GPClassificationModel
import torch

# 4维刺激空间（亮度、对比度、饱和度、色调）
model = GPClassificationModel(
    lb=torch.tensor([0, 0, 0, 0]),
    ub=torch.tensor([1, 1, 1, 1]),
    dim=4
)

# 配置采集函数
acqf = EURAnovaMultiAcqf(
    model,
    # 假设：亮度和对比度有交互，饱和度和色调有交互
    enable_main=True,
    interaction_pairs=[(0, 1), (2, 3)],  # 亮度-对比度, 饱和度-色调
    interaction_triplets=[(0, 1, 2)],    # 探索三者联合效应

    # 混合变量类型
    variable_types={
        0: 'continuous',  # 亮度
        1: 'continuous',  # 对比度
        2: 'categorical', # 饱和度（低/中/高）
        3: 'continuous'   # 色调
    },

    # 实验预算：30次trial
    total_budget=30,
    lambda_2=1.0,
    lambda_3=0.5,

    debug_components=True
)

# 生成候选点
X_candidates = torch.rand(100, 4)

# 评估并选择最佳点
scores = acqf(X_candidates)
best_idx = scores.argmax()
next_trial = X_candidates[best_idx]

print(f"推荐下一个试验点: {next_trial}")
acqf.print_diagnostics()
```

---

## 📖 参考文献

- Owen et al. (2021). "Adaptive Experimentation in Psychophysics"
- Montgomery (2017). "Design and Analysis of Experiments"
- Box & Draper (1987). "Empirical Model-Building and Response Surfaces"

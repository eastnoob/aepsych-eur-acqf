# ⭐ AEPsych 配置指南：EURAnovaPairAcqf

> **文件版本**: 2025-11-04  
> **采集函数**: `EURAnovaPairAcqf` (eur_anova_pair.py)  
> **适用框架**: AEPsych 动态实验平台

---

## 📋 目录

1. [快速开始](#快速开始)
2. [完整参数表](#完整参数表)
3. [参数详细说明](#参数详细说明)
4. [INI 配置示例](#ini-配置示例)
5. [参数映射机制](#参数映射机制)
6. [常见配置场景](#常见配置场景)
7. [调试与诊断](#调试与诊断)

---

## 🚀 快速开始

### 最小化配置（使用全部默认值）

```ini
[common]
parnames = [color, layout, size, opacity]
stimuli_per_trial = 1
outcome_types = [ordinal]
strategy_names = [init_strat, opt_strat]

[init_strat]
min_asks = 10
generator = SobolGenerator

[opt_strat]
min_asks = 20
generator = OptimizeAcqfGenerator
acqf = EURAnovaPairAcqf

# 👇 EURAnovaPairAcqf 不需要任何参数，全部使用默认值
[EURAnovaPairAcqf]
# 空白即可！所有参数都有合理默认值
```

### 推荐配置（20-30次试验预算）

```ini
[EURAnovaPairAcqf]
# 实验预算自适应（最简单，推荐！）
total_budget = 25

# 交互对（必须手动指定）
interaction_pairs = 0,1; 0,2; 1,2

# 变量类型（推荐显式指定）
variable_types_list = categorical, categorical, integer, continuous
```

---

## 📊 完整参数表

### 必须设置的参数

| 参数名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `interaction_pairs` | str/list | 交互对列表（强烈建议设置） | `"0,1; 2,3"` |

> ⚠️ **注意**: 虽然技术上所有参数都有默认值，但 `interaction_pairs` 强烈建议手动指定，否则无法利用交互效应探索功能。

### 有默认值的参数（按功能分类）

#### 1️⃣ 核心权重参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `gamma` | float | 0.3 | [0, 1] | 信息/覆盖初始融合权重 |
| `main_weight` | float | 1.0 | (0, ∞) | 主效应权重（严格遵循设计公式应为1.0） |

#### 2️⃣ 动态交互效应权重（λ_t 自适应）

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `use_dynamic_lambda` | bool | True | - | 是否启用动态交互权重 |
| `tau1` | float | 0.7 | [0, 1] | r_t 上阈值（高于此值降低交互权重） |
| `tau2` | float | 0.3 | [0, 1] | r_t 下阈值（低于此值提高交互权重） |
| `lambda_min` | float | 0.1 | [0, ∞) | 最小交互权重（参数已收敛时） |
| `lambda_max` | float | 1.0 | [0, ∞) | 最大交互权重（参数不确定时） |

> 💡 **约束**: `tau1 > tau2`, `lambda_max >= lambda_min`

#### 3️⃣ 动态覆盖权重（γ_t 自适应）

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `use_dynamic_gamma` | bool | True | - | 是否启用动态覆盖权重 |
| `gamma_max` | float | 0.5 | [0, 1] | 最大覆盖权重（早期探索阶段） |
| `gamma_min` | float | 0.05 | [0, 1] | 最小覆盖权重（后期精细化阶段） |
| `tau_n_min` | int | 3 | [1, ∞) | 样本数下限阈值 |
| `tau_n_max` | int | 25 | [1, ∞) | 样本数上限阈值 |

> 💡 **约束**: `gamma_max >= gamma_min`, `tau_n_max > tau_n_min`

#### 4️⃣ 局部扰动参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `local_jitter_frac` | float | 0.1 | [0, 1] | 局部扰动尺度（相对于数据范围） |
| `local_num` | int | 4 | [1, ∞) | 每个维度的局部采样数量 |

#### 5️⃣ 变量类型配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `variable_types` | dict | None | 变量类型字典（Python代码中使用） |
| `variable_types_list` | str/list | None | 变量类型列表（INI配置中使用） |

> 📝 **说明**: 
> - `variable_types`: 字典格式 `{0: "categorical", 1: "continuous"}`
> - `variable_types_list`: 字符串/列表格式 `"categorical, integer, continuous"`
> - 如果都不指定，会尝试从模型的 `transforms` 自动推断

#### 6️⃣ 覆盖度计算

| 参数名 | 类型 | 默认值 | 可选值 | 说明 |
|--------|------|--------|--------|------|
| `coverage_method` | str | "min_distance" | min_distance, mean_distance, median_distance | Gower距离聚合方法 |

#### 7️⃣ 实验预算助手（✨ 推荐使用）

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `total_budget` | int | None | 总采样次数（自动配置 tau_n_max/gamma_min） |

> ✨ **自动配置规则**:
> - `tau_n_max = int(total_budget * 0.7)` (预算的70%)
> - `gamma_min = 0.05` (如果 budget < 30) 或 `0.1` (如果 budget >= 30)
> - 只在用户未手动配置时生效（手动配置优先）

#### 8️⃣ 调试选项

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `debug_components` | bool/str | False | 是否缓存中间计算结果（用于调试） |
| `random_seed` | int | 42 | 随机种子（确保可重复性） |

---

## 📖 参数详细说明

### 1. 交互对配置 (`interaction_pairs`)

**支持的格式**:

```ini
# 格式1: 分号分隔
interaction_pairs = 0,1; 2,3; 1,2

# 格式2: 竖线分隔
interaction_pairs = 0|1; 2|3; 1|2

# 格式3: 混合分隔符
interaction_pairs = "0,1", "2,3", "1,2"
```

**Python代码中的格式**:

```python
# 列表格式
interaction_pairs = [(0, 1), (2, 3), (1, 2)]

# 字符串格式（同INI）
interaction_pairs = "0,1; 2,3; 1,2"
```

**自动去重**: 输入 `"0,1; 1,0; 0,1"` 会自动去重为 `[(0,1)]`

**索引验证**: 
- ✅ 首次调用 `forward()` 时自动验证索引范围
- ⚠️ 越界索引会触发警告并自动过滤
- 示例: 如果维度=4，`[(0,1), (3,5)]` 会过滤为 `[(0,1)]`

---

### 2. 变量类型配置 (`variable_types_list`)

**支持的变量类型**:

| 类型标记 | 匹配关键字 | 处理方式 |
|---------|-----------|---------|
| `categorical` | cat, categorical | 从训练数据unique值采样 |
| `integer` | int, integer | 高斯扰动后舍入+夹值 |
| `continuous` | cont, continuous, float, real | 标准高斯扰动 |

**INI配置格式**:

```ini
# 格式1: 逗号分隔字符串
variable_types_list = categorical, integer, continuous, categorical

# 格式2: 列表格式（AEPsych会解析）
variable_types_list = [categorical, integer, continuous, categorical]

# 格式3: 分号分隔
variable_types_list = categorical; integer; continuous; categorical
```

**自动推断规则**:

```
优先级: variable_types (dict) > variable_types_list (解析) > 自动推断 (transforms)
```

**推断示例**:

```python
# 如果模型有 Categorical transform 在维度0
# 会自动推断 variable_types = {0: "categorical"}

# 如果模型有 Round transform 在维度1
# 会自动推断 variable_types = {1: "integer"}
```

---

### 3. 动态权重机制

#### λ_t（交互效应权重）动态公式

```
λ_t(r_t) = {
    λ_min,                                    if r_t > τ_1
    λ_min + (λ_max - λ_min)·(τ_1-r_t)/(τ_1-τ_2), if τ_2 ≤ r_t ≤ τ_1
    λ_max,                                    if r_t < τ_2
}

其中 r_t = (1/|J|)·∑ Var[θ_j|D_t] / Var[θ_j|D_0]
```

**直觉理解**:
- **早期** (r_t ≈ 1, 参数不确定): λ_t ≈ λ_min，聚焦主效应
- **后期** (r_t → 0, 参数收敛): λ_t → λ_max，探索交互效应

#### γ_t（覆盖权重）动态公式

```
γ_base = {
    γ_max,                                     if n < τ_n_min
    γ_max - (γ_max - γ_min)·(n-τ_n_min)/(τ_n_max-τ_n_min), if τ_n_min ≤ n ≤ τ_n_max
    γ_min,                                     if n > τ_n_max
}

γ_t = γ_base · adjustment(r_t)  # 可选：基于r_t微调±20%
```

**直觉理解**:
- **早期** (n < tau_n_min): γ_t = γ_max，重视覆盖（探索）
- **后期** (n > tau_n_max): γ_t = γ_min，重视信息（开发）

---

### 4. 实验预算助手 (`total_budget`)

**使用场景**: 你知道总共要做多少次试验（例如 20次），让系统自动配置最优参数

**配置示例**:

```ini
[EURAnovaPairAcqf]
total_budget = 20
```

**自动效果**:

```
total_budget = 20 →
  - tau_n_max = 14 (20 * 0.7)
  - gamma_min = 0.05 (因为 20 < 30)

total_budget = 50 →
  - tau_n_max = 35 (50 * 0.7)
  - gamma_min = 0.1 (因为 50 >= 30)
```

**优先级**: 如果你手动设置了 `tau_n_max` 或 `gamma_min`，`total_budget` 不会覆盖它们

---

### 5. 调试工具

#### 启用调试模式

```ini
[EURAnovaPairAcqf]
debug_components = true
```

**效果**: 每次调用 `forward()` 后，会缓存以下中间结果：

```python
acqf._last_main      # 主效应贡献
acqf._last_pair      # 交互效应贡献
acqf._last_info      # 信息项
acqf._last_cov       # 覆盖项
```

#### 获取诊断信息

```python
# 方法1: 获取诊断字典
diag = acqf.get_diagnostics()
print(f"当前 λ_t = {diag['lambda_t']:.3f}")
print(f"当前 γ_t = {diag['gamma_t']:.3f}")
print(f"训练样本数 = {diag['n_train']}")

# 方法2: 打印格式化报告
acqf.print_diagnostics()              # 简要报告
acqf.print_diagnostics(verbose=True)  # 详细报告（包含数组）
```

**示例输出**:

```
======================================================================
EURAnovaPairAcqf 诊断信息
======================================================================

【动态权重状态】
  λ_t (交互权重) = 0.8523  (范围: [0.1, 1.0])
  γ_t (覆盖权重) = 0.1250  (范围: [0.05, 0.5])

【模型状态】
  训练样本数: 18
  转向阈值: tau_n_min=3, tau_n_max=25
  模型已拟合: 是

【交互对配置】
  交互对数量: 3
  交互对: (0,1), (0,2), (1,2)

【效应贡献】(最后一次 forward() 调用)
  主效应总和: mean=0.1234, std=0.0456
  交互效应总和: mean=0.0789, std=0.0234
  信息项: mean=0.5678, std=0.1234
  覆盖项: mean=0.3456, std=0.0987
======================================================================
```

---

## 🔧 INI 配置示例

### 示例1: 序数响应 + 混合变量类型（推荐配置）

```ini
[common]
parnames = [color, layout, size, opacity]
stimuli_per_trial = 1
outcome_types = [ordinal]
strategy_names = [init_strat, opt_strat]

# 变量定义
[color]
par_type = categorical
choices = [red, blue, green, yellow]

[layout]
par_type = categorical
choices = [grid, list, flow]

[size]
par_type = integer
lower_bound = 10
upper_bound = 20

[opacity]
par_type = continuous
lower_bound = 0.0
upper_bound = 1.0

# 采样策略
[init_strat]
min_asks = 10
generator = SobolGenerator

[opt_strat]
min_asks = 20
generator = OptimizeAcqfGenerator
acqf = EURAnovaPairAcqf

# 采集函数配置
[EURAnovaPairAcqf]
# 实验预算（最简单！）
total_budget = 30

# 交互对（必须指定）
interaction_pairs = 0,1; 0,2; 1,3

# 变量类型（推荐显式指定）
variable_types_list = categorical, categorical, integer, continuous

# 其他参数使用默认值即可
```

---

### 示例2: 连续响应 + 高维问题

```ini
[common]
parnames = [x1, x2, x3, x4, x5, x6]
stimuli_per_trial = 1
outcome_types = [continuous]
strategy_names = [init_strat, opt_strat]

[x1]
par_type = continuous
lower_bound = 0.0
upper_bound = 1.0

[x2]
par_type = continuous
lower_bound = 0.0
upper_bound = 1.0

[x3]
par_type = integer
lower_bound = 1
upper_bound = 10

[x4]
par_type = categorical
choices = [A, B, C, D]

[x5]
par_type = continuous
lower_bound = -5.0
upper_bound = 5.0

[x6]
par_type = integer
lower_bound = 0
upper_bound = 100

[init_strat]
min_asks = 20
generator = SobolGenerator

[opt_strat]
min_asks = 50
generator = OptimizeAcqfGenerator
acqf = EURAnovaPairAcqf

[EURAnovaPairAcqf]
# 高维问题配置
total_budget = 70
interaction_pairs = 0,1; 0,3; 1,2; 2,4; 3,5

# 变量类型
variable_types_list = continuous, continuous, integer, categorical, continuous, integer

# 调整动态权重
tau_n_max = 50        # 更晚转向精细化
gamma_min = 0.08      # 保持一定覆盖

# 增加局部采样
local_num = 6         # 更多局部探索
```

---

### 示例3: 纯分类变量 + 小样本

```ini
[common]
parnames = [color, shape, pattern, texture]
stimuli_per_trial = 1
outcome_types = [binary]
strategy_names = [init_strat, opt_strat]

[color]
par_type = categorical
choices = [red, blue, green]

[shape]
par_type = categorical
choices = [circle, square, triangle]

[pattern]
par_type = categorical
choices = [solid, stripe, dot]

[texture]
par_type = categorical
choices = [smooth, rough, glossy]

[init_strat]
min_asks = 8
generator = SobolGenerator

[opt_strat]
min_asks = 12
generator = OptimizeAcqfGenerator
acqf = EURAnovaPairAcqf

[EURAnovaPairAcqf]
# 小样本配置
total_budget = 20

# 所有两两交互（因为纯分类）
interaction_pairs = 0,1; 0,2; 0,3; 1,2; 1,3; 2,3

# 变量类型
variable_types_list = categorical, categorical, categorical, categorical

# 保持较高覆盖
gamma = 0.4
gamma_min = 0.1

# 降低局部采样（分类变量采样受限）
local_num = 3
```

---

### 示例4: 调试模式

```ini
[EURAnovaPairAcqf]
# 启用所有诊断功能
debug_components = true
random_seed = 12345

# 其他配置...
total_budget = 25
interaction_pairs = 0,1; 1,2
variable_types_list = categorical, continuous, integer
```

**Python脚本中获取诊断**:

```python
# 在实验运行后
from aepsych.server import AEPsychServer

server = AEPsychServer()
# ... 运行实验 ...

# 获取采集函数
strat = server.strat
acqf = strat.generator.acqf

# 打印诊断
acqf.print_diagnostics(verbose=True)

# 或导出诊断数据
import json
diag = acqf.get_diagnostics()
with open('diagnostics.json', 'w') as f:
    json.dump(diag, f, indent=2, default=str)
```

---

## 🔗 参数映射机制

### INI 到 Python 的映射流程

```
┌─────────────────────────────────────────────────────┐
│ 1. INI 文件解析                                       │
│    config.ini → ConfigParser → dict                  │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│ 2. AEPsych Config 对象                               │
│    Config.from_ini() → Config 对象                   │
│    - 解析 [section] 为字典                           │
│    - 类型转换（str→int/float/bool/list）            │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│ 3. Strategy 构建                                     │
│    Strategy.from_config(config) → Strategy 对象      │
│    - 读取 [opt_strat] 的 acqf 参数                   │
│    - 查找对应的 [EURAnovaPairAcqf] section          │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│ 4. 采集函数实例化                                     │
│    acqf_class(**config_dict)                         │
│    - 将 [EURAnovaPairAcqf] 字典解包为关键字参数      │
│    - Python __init__() 接收参数                      │
└─────────────────────────────────────────────────────┘
```

### 类型转换规则

AEPsych Config 会自动进行以下类型转换：

| INI 中的字符串 | 转换后类型 | 示例 |
|---------------|-----------|------|
| `true`, `True`, `yes`, `1` | bool (True) | `use_dynamic_lambda = true` |
| `false`, `False`, `no`, `0` | bool (False) | `debug_components = false` |
| `3.14`, `0.5` | float | `gamma = 0.3` |
| `42`, `100` | int | `tau_n_max = 25` |
| `0,1; 2,3` | str (待解析) | `interaction_pairs = "0,1; 2,3"` |
| `[cat, int, cont]` | list | `variable_types_list = [categorical, integer, continuous]` |

### 自定义参数名映射（可选）

如果需要在INI中使用不同的参数名，可以在采集函数类中添加 `@classmethod` 来处理：

```python
class EURAnovaPairAcqf(AcquisitionFunction):
    @classmethod
    def from_config(cls, config: Config, section: str = "EURAnovaPairAcqf"):
        """从 Config 对象构建采集函数（支持自定义映射）"""
        params = config[section]
        
        # 自定义映射（例如支持别名）
        if 'budget' in params:
            params['total_budget'] = params.pop('budget')
        
        if 'pairs' in params:
            params['interaction_pairs'] = params.pop('pairs')
        
        # 调用标准构造函数
        return cls(**params)
```

**使用别名的INI配置**:

```ini
[EURAnovaPairAcqf]
budget = 30           # 别名，映射到 total_budget
pairs = 0,1; 2,3     # 别名，映射到 interaction_pairs
```

> ⚠️ **注意**: 当前版本不支持别名，上面仅为示例说明机制

---

## 🎯 常见配置场景

### 场景1: 我的实验预算很少（<20次）

```ini
[EURAnovaPairAcqf]
total_budget = 15

# 更早转向精细化
tau_n_max = 10       # 约为 budget * 0.67

# 保持较高覆盖
gamma_min = 0.08

# 减少交互对（聚焦关键交互）
interaction_pairs = 0,1; 1,2
```

---

### 场景2: 我的实验预算充足（>50次）

```ini
[EURAnovaPairAcqf]
total_budget = 80

# 更晚转向精细化
tau_n_max = 60       # 约为 budget * 0.75

# 降低后期覆盖
gamma_min = 0.1

# 探索更多交互对
interaction_pairs = 0,1; 0,2; 0,3; 1,2; 1,3; 2,3
```

---

### 场景3: 我主要关心主效应

```ini
[EURAnovaPairAcqf]
# 降低交互权重范围
lambda_min = 0.05
lambda_max = 0.5

# 或完全禁用动态交互
use_dynamic_lambda = false
main_weight = 1.5         # 提高主效应权重

# 最小化交互对
interaction_pairs = 0,1   # 只设置1-2个关键交互
```

---

### 场景4: 我想探索复杂交互

```ini
[EURAnovaPairAcqf]
# 提高交互权重范围
lambda_min = 0.3
lambda_max = 2.0

# 更早开始探索交互
tau1 = 0.8
tau2 = 0.4

# 设置所有可能的交互对
interaction_pairs = 0,1; 0,2; 0,3; 1,2; 1,3; 2,3; ...
```

---

### 场景5: 我有许多分类变量

```ini
[EURAnovaPairAcqf]
# 确保指定变量类型（重要！）
variable_types_list = categorical, categorical, categorical, integer

# 增加局部采样（分类变量需要更多探索）
local_num = 6

# 使用较高的覆盖权重
gamma = 0.4
gamma_min = 0.15

# 覆盖方法使用最小距离
coverage_method = min_distance
```

---

### 场景6: 我想确保结果可重复

```ini
[EURAnovaPairAcqf]
random_seed = 42          # 固定随机种子

# 在 [common] 也设置种子
[common]
random_seed = 42
```

---

## ⚠️ 常见错误与警告

### 错误1: 交互对索引越界

**错误信息**:
```
UserWarning: 交互对包含越界索引（维度=4）：[(3, 5), (0, 6)]，已自动过滤。
请检查 interaction_pairs 配置是否正确。
```

**原因**: 交互对中的索引超过了变量维度

**解决方法**:
```ini
# 错误配置（假设只有4个变量：0,1,2,3）
interaction_pairs = 0,1; 3,5  # ❌ 索引5不存在

# 正确配置
interaction_pairs = 0,1; 2,3  # ✅ 所有索引 < 4
```

---

### 错误2: 序数模型但无cutpoints

**警告信息**:
```
UserWarning: 检测到序数似然模型（OrdinalLikelihood），但无法获取cutpoints。
这可能表示配置错误。信息增益计算将退化为方差指标（Var[p̂]）。
建议检查：
  1. OrdinalLikelihood是否正确初始化（需要n_levels参数）
  2. 模型是否已经过训练（cutpoints在训练时学习）
  3. cutpoints属性是否可访问
当前配置下将使用方差指标继续计算。
```

**原因**: 使用了 `outcome_types = [ordinal]`，但模型配置不完整

**解决方法**:

```ini
# 确保在 [common] 或模型配置中正确设置
[GPClassificationModel]
inducing_size = 100
mean_covar_factory = default_mean_covar_factory
n_levels = 5          # ✅ 必须指定等级数
```

**注意**: 这只是**警告**，不会阻止运行，程序会自动降级为使用方差指标

---

### 错误3: 参数约束冲突

**错误信息**:
```
ValueError: tau1 must be > tau2 for proper dynamic lambda weighting, 
got tau1=0.3, tau2=0.5
```

**原因**: 违反了参数约束

**解决方法**:
```ini
# 错误配置
tau1 = 0.3
tau2 = 0.5  # ❌ tau2 > tau1

# 正确配置
tau1 = 0.7  # ✅ tau1 > tau2
tau2 = 0.3
```

**所有约束**:
- `tau1 > tau2`
- `lambda_max >= lambda_min`
- `gamma_max >= gamma_min`
- `tau_n_max > tau_n_min`

---

### 错误4: 分类变量值预计算失败

**警告信息**:
```
UserWarning: 预计算分类值失败的维度: [(2, 'index out of range [0, 4)')], 
这些维度将保持原值（无局部探索）
```

**原因**: `variable_types_list` 中标记为分类的维度超出实际范围

**解决方法**:

```ini
# 错误配置（只有4个变量）
variable_types_list = categorical, categorical, categorical, categorical, categorical
# ❌ 第5个分类变量不存在

# 正确配置（4个变量）
variable_types_list = categorical, categorical, integer, continuous
# ✅ 正好4个类型
```

---

### 错误5: main_weight 偏离设计值

**警告信息**:
```
UserWarning: main_weight=2.0 deviates from design formula (should be 1.0). 
This may be acceptable for specific scenarios but changes the balance.
```

**原因**: `main_weight` 不是推荐的默认值 1.0

**说明**: 这是一个**提示**，不是错误。设计公式建议 `main_weight = 1.0`，但你可以根据需要调整

**如果你确实需要调整**:

```ini
[EURAnovaPairAcqf]
main_weight = 1.5  # 可以设置，但会收到提示
# 你可以忽略这个警告，如果这是你有意的选择
```

---

## 📚 参考资源

### 相关文档

- **调试工具指南**: `DEBUG_TOOLS_GUIDE.md`
- **修改计划**: `MODIFICATION_PLAN.md`
- **源代码**: `eur_anova_pair.py`
- **测试文件**: `test_modifications.py`

### AEPsych 官方文档

- **配置文件格式**: https://aepsych.org/docs/configs
- **采集函数**: https://aepsych.org/docs/acquisition
- **变量类型**: https://aepsych.org/docs/transforms

---

## 🆘 常见问题 FAQ

### Q1: 我必须设置 `interaction_pairs` 吗？

**A**: 技术上不是必须的（默认为空列表），但**强烈推荐**设置。如果不设置，采集函数将只考虑主效应，无法利用交互效应探索的优势。

---

### Q2: `variable_types_list` 和 `variable_types` 有什么区别？

**A**: 
- `variable_types_list`: 字符串/列表格式，适合在 **INI 配置**中使用
- `variable_types`: 字典格式，适合在 **Python 代码**中使用

两者功能相同，只是格式不同。INI 中推荐用 `variable_types_list`。

---

### Q3: 如何选择 `interaction_pairs`？

**A**: 基于领域知识选择：

1. **全组合**: 如果变量不多（d≤5），可以设置所有两两组合
   ```python
   [(i,j) for i in range(d) for j in range(i+1, d)]
   ```

2. **关键交互**: 基于领域知识，只设置可能有交互的变量对
   ```ini
   # 例如：颜色和布局可能交互，但字体大小和透明度不太可能
   interaction_pairs = 0,1; 0,2
   ```

3. **分阶段**: 先探索主效应（少设置交互对），再根据结果添加

---

### Q4: `total_budget` 和手动设置 `tau_n_max`/`gamma_min` 冲突怎么办？

**A**: **手动设置优先**。`total_budget` 只在对应参数使用默认值时生效。

```ini
# 场景1: total_budget 生效
[EURAnovaPairAcqf]
total_budget = 30
# tau_n_max 会被设置为 21 (30 * 0.7)

# 场景2: 手动设置优先
[EURAnovaPairAcqf]
total_budget = 30
tau_n_max = 40      # ✅ 使用 40，忽略 total_budget 的自动计算
```

---

### Q5: 如何调试采集函数不按预期工作？

**A**: 启用调试模式并查看诊断信息：

```ini
[EURAnovaPairAcqf]
debug_components = true
```

```python
# Python脚本中
acqf.print_diagnostics(verbose=True)

# 检查：
# 1. 动态权重是否合理（λ_t, γ_t）
# 2. 训练样本数是否正确
# 3. 交互对配置是否生效
# 4. 效应贡献是否符合预期
```

---

### Q6: 警告信息会影响实验运行吗？

**A**: **不会**。Python 的 `warnings.warn()` 只输出提示信息，不会中断程序。

- ✅ 程序继续正常运行
- ⚠️ 但建议查看警告并修复配置问题
- 🛑 只有 `raise ValueError()` 等错误会中断程序

---

### Q7: 我的变量全是分类的，如何配置？

**A**: 需要特别注意以下几点：

```ini
[EURAnovaPairAcqf]
# 1. 明确标记所有变量为分类
variable_types_list = categorical, categorical, categorical, categorical

# 2. 增加局部采样数（分类变量采样受限）
local_num = 5

# 3. 保持较高覆盖权重
gamma = 0.4
gamma_min = 0.15

# 4. 覆盖方法建议使用最小距离
coverage_method = min_distance
```

---

### Q8: 如何确保实验结果可重复？

**A**: 设置固定的随机种子：

```ini
[common]
random_seed = 42

[EURAnovaPairAcqf]
random_seed = 42
```

同时确保：
- Python 环境相同
- 数据加载顺序相同
- AEPsych 版本相同

---

### Q9: 参数太多了，有推荐的"懒人配置"吗？

**A**: 有！使用 `total_budget` + `interaction_pairs` 即可：

```ini
[EURAnovaPairAcqf]
total_budget = 30                    # 你的总试验次数
interaction_pairs = 0,1; 0,2; 1,2   # 关键交互对
variable_types_list = categorical, continuous, integer  # 变量类型

# 其他全部使用默认值！
```

这会自动配置：
- ✅ `tau_n_max` = 21 (70% 的预算)
- ✅ `gamma_min` = 0.05 (因为 budget < 30)
- ✅ 其他所有参数使用优化过的默认值

---

### Q10: 如何在 Python 代码中直接使用（不通过 INI）？

**A**: 直接实例化类：

```python
from extensions.dynamic_eur_acquisition import EURAnovaPairAcqf
from botorch.models import SingleTaskGP

# 1. 训练模型
model = SingleTaskGP(train_X, train_Y)
model.train()

# 2. 创建采集函数
acqf = EURAnovaPairAcqf(
    model=model,
    total_budget=30,
    interaction_pairs=[(0, 1), (0, 2), (1, 2)],
    variable_types={0: "categorical", 1: "continuous", 2: "integer"},
    debug_components=True
)

# 3. 评估候选点
candidates = torch.rand(10, 3)
scores = acqf(candidates)
best_idx = scores.argmax()
```

---

## 📝 版本信息

- **文档版本**: 1.0.0
- **采集函数版本**: eur_anova_pair.py (2025-11-04)
- **最后更新**: 2025-11-04
- **维护者**: AEPsych Dynamic EUR Acquisition Team

---

## 📄 许可证

本文档遵循 AEPsych 主项目的许可证。

---

**💡 提示**: 如果你在配置中遇到问题，可以：

1. 查看 `DEBUG_TOOLS_GUIDE.md` 了解调试方法
2. 检查 `MODIFICATION_PLAN.md` 了解设计原理
3. 运行 `test_modifications.py` 验证安装
4. 在 GitHub Issues 中提问

**🎯 记住**: 最简配置只需要 3 行：

```ini
[EURAnovaPairAcqf]
total_budget = 30
interaction_pairs = 0,1; 1,2
variable_types_list = categorical, continuous, integer
```

**其他全部用默认值！** ✨

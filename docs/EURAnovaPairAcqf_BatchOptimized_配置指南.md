# EURAnovaPairAcqf_BatchOptimized 配置指南

## 概述

`EURAnovaPairAcqf_BatchOptimized` 是基于期望效用理论与ANOVA分解的高阶采集函数，支持混合变量类型（分类/整数/连续）和动态权重调整。该版本经过批量性能优化，显著降低了模型评估次数。

## 配置区域与关键字

在AEPsych的`.ini`配置文件中，需要在对应策略部分下添加 `[EURAnovaPairAcqf_BatchOptimized]` 配置区域。

### 必需配置区域

#### `[EURAnovaPairAcqf_BatchOptimized]`

这是主要的配置区域，包含所有采集函数参数。

**必须的关键字：**

- 无绝对必需的关键字，所有参数都有默认值

### 配置参数详解

#### 信息/覆盖融合参数

| 关键字 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `gamma` | float | 0.3 | 否 | 信息/覆盖平衡权重。范围[0,1]，越高越重视覆盖探索 |

#### 主/交互效应权重参数

| 关键字 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `main_weight` | float | 1.0 | 否 | 主效应权重，严格遵循设计公式α_info = (1/\|J\|)·∑Δ_j + λ_t·(1/\|I\|)·∑Δ_ij |
| `pair_weight` | float | 1.0 | 否 | 交互效应基础权重 |

#### 动态权重参数 (λ_t 交互效应自适应)

| 关键字 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `use_dynamic_lambda` | bool | True | 否 | 是否启用动态λ_t调整 |
| `tau1` | float | 0.7 | 否 | r_t 上阈值，高于此值降低交互权重 |
| `tau2` | float | 0.3 | 否 | r_t 下阈值，低于此值提高交互权重 |
| `lambda_min` | float | 0.1 | 否 | 最小交互权重（参数已收敛时） |
| `lambda_max` | float | 1.0 | 否 | 最大交互权重（参数不确定时） |

#### 交互对配置

| 关键字 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `interaction_pairs` | string | None | 否 | 交互对列表，支持格式："0,1;2,3" 或 [(0,1), (2,3)] |

#### 局部扰动参数

| 关键字 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `local_jitter_frac` | float | 0.1 | 否 | 局部扰动幅度比例 |
| `local_num` | int | 4 | 否 | 每个维度的局部样本数 |

#### 变量类型配置

| 关键字 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `variable_types_list` | string | None | 否 | 变量类型列表，格式："categorical, integer, integer, continuous, categorical, integer" |
| `coverage_method` | string | "min_distance" | 否 | 覆盖计算方法 |

#### 动态γ_t参数

| 关键字 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `use_dynamic_gamma` | bool | True | 否 | 是否启用动态γ_t调整 |
| `gamma_max` | float | 0.5 | 否 | 最大γ值（样本少时重视覆盖） |
| `gamma_min` | float | 0.1 | 否 | 最小γ值（样本多时重视信息） |
| `tau_n_min` | int | 3 | 否 | 样本数下阈值 |
| `tau_n_max` | int | 40 | 否 | 样本数上阈值 |

#### 其他参数

| 关键字 | 类型 | 默认值 | 必需 | 说明 |
|--------|------|--------|------|------|
| `random_seed` | int | 42 | 否 | 随机种子，确保确定性行为 |
| `debug_components` | bool/str | False | 否 | 是否启用调试模式 |

## 配置示例

### 基础配置

```ini
[EURAnovaPairAcqf_BatchOptimized]
gamma = 0.3
use_dynamic_lambda = true
interaction_pairs = "0,1;2,3;4,5"
local_jitter_frac = 0.08
local_num = 4
variable_types_list = categorical, integer, integer, continuous, categorical, integer
```

### 高级配置（带动态权重调整）

```ini
[EURAnovaPairAcqf_BatchOptimized]
# 信息覆盖平衡
gamma = 0.3
use_dynamic_gamma = true
gamma_max = 0.6
gamma_min = 0.1
tau_n_min = 5
tau_n_max = 60

# 主交互权重
main_weight = 0.6
pair_weight = 1.0

# 动态交互权重
use_dynamic_lambda = true
tau1 = 0.7
tau2 = 0.3
lambda_min = 0.1
lambda_max = 1.0

# 交互对
interaction_pairs = "0,1;2,3;4,5"

# 局部扰动
local_jitter_frac = 0.08
local_num = 4

# 变量类型
variable_types_list = categorical, integer, integer, continuous, categorical, integer

# 调试
debug_components = false
```

## 参数验证规则

系统会对以下参数进行验证：

1. **权重参数**：`main_weight > 0`
2. **动态权重阈值**：`tau1 > tau2`
3. **λ范围**：`lambda_max >= lambda_min`
4. **γ范围**：`gamma_max >= gamma_min`
5. **样本数阈值**：`tau_n_max > tau_n_min`

违反这些规则将抛出 `ValueError`。

## 设计公式

采集函数α(x)计算公式：

```
α(x) = α_info(x) + γ_t · COV(x)

其中：
α_info(x) = (1/|J|)·∑_j Δ_j + λ_t(r_t)·(1/|I|)·∑_(i,j) Δ_ij

λ_t = f(r_t)  # 参数方差比的分段函数
γ_t = g(n, r_t)  # 样本数与参数方差的联合函数
```

## 注意事项

1. **变量类型**：强烈推荐显式指定`variable_types_list`，否则系统会尝试从模型变换自动推断
2. **交互对**：只对感兴趣的变量对指定交互，避免计算开销过大
3. **动态调整**：默认启用动态权重调整，通常不需要手动调整阈值
4. **性能**：批量优化版本在高维情况下性能提升显著（~21x加速比）
5. **兼容性**：该函数仅支持`q=1`的候选点批次

## 故障排除

- **分类变量非法值**：确保训练数据包含所有合法分类值，或使用`variable_types_list`显式指定
- **性能问题**：减少`local_num`或交互对数量
- **数值不稳定**：检查模型后验方差计算，必要时调整`local_jitter_frac`

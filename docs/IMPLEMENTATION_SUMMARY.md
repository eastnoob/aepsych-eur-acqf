# Dynamic EUR Acquisition Function - 实现总结

## 项目概述

已成功实现一个完整的主动学习采集函数（Dynamic EUR Acquisition Function），用于 AEPsych 框架扩展。该实现位于 `extensions/dynamic_eur_acquisition` 目录中。

## 已完成的功能

### ✅ 1. 核心模块

#### `acquisition_function.py` - 主采集函数

- 实现了 `DynamicEURAcquisitionFunction` 类
- 支持信息增益（EUR）计算，包括主效应和交互效应
- 实现了空间覆盖度计算
- 动态权重机制 λ_t(r_t)，根据方差减少自动调整
- 支持从 INI 配置文件加载参数
- 提供 `fit()`, `__call__()`, `select_next()` 等完整 API

#### `gower_distance.py` - Gower 距离计算

- 支持混合类型变量（连续型和分类型）
- 实现了单点和批量 Gower 距离计算
- 提供空间覆盖度评分函数
- 支持多种覆盖度计算方法（min_distance, mean_distance, median_distance）

#### `gp_variance.py` - 高斯过程方差计算

- 实现了 `GPVarianceCalculator` 类
- 使用贝叶斯线性回归估计参数方差
- 支持主效应和交互效应的方差计算
- 提供方差减少量计算
- 使用 Cholesky 分解保证数值稳定性

### ✅ 2. 配置系统

#### `config_template.ini` - 配置模板

- 提供所有参数的说明和默认值
- 包含详细的注释

#### `config_example.ini` - 配置示例

- 展示如何配置交互项
- 提供实际使用场景的参数设置

### ✅ 3. 文档

#### `README.md` - 快速入门

- 功能概述和特性列表
- 快速开始示例
- 目录结构说明
- 主要参数说明

#### `doc/README.md` - 完整文档

- 数学公式详解
- 完整的 API 参考
- 高级使用示例
- 配置指南
- 常见问题解答

### ✅ 4. 示例和测试

#### `example_usage.py` - 使用示例

- 示例 1：基本使用（默认参数）
- 示例 2：使用交互项
- 示例 3：从配置文件加载
- 示例 4：主动学习循环
- 示例 5：混合变量类型

#### `test/test_acquisition_function.py` - 单元测试

- Gower 距离测试
- 空间覆盖度测试
- GP 方差计算测试
- 采集函数集成测试
- 主动学习循环测试

#### `simple_test.py` - 简单验证测试

- 快速验证模块是否正常工作

## 数学实现

### 采集分数公式

$$\alpha(x; D_t) = \alpha_{\text{info}}(x; D_t) + \alpha_{\text{cov}}(x; D_t)$$

### 信息增益部分

$$\alpha_{\text{info}}(x; D_t) = \frac{1}{|\mathcal{J}|} \sum_{j \in \mathcal{J}} \Delta \text{Var}_{GP}[\theta_j] + \lambda_t \cdot \frac{1}{|\mathcal{I}|} \sum_{(j,k) \in \mathcal{I}} \Delta \text{Var}_{GP}[\theta_{jk}]$$

### 空间覆盖部分

$$\alpha_{\text{cov}}(x; D_t) = \gamma \cdot \min_{x' \in D_t} d_{\text{Gower}}(x, x')$$

### 动态权重

$$\lambda_t(r_t) = \begin{cases}
\lambda_{\min}, & r_t > \tau_1 \\
\text{linear interpolation}, & \tau_2 \leq r_t \leq \tau_1 \\
\lambda_{\max}, & r_t < \tau_2
\end{cases}$$

## 主要特性

### 1. 灵活配置
- ✅ 支持默认参数（无需配置即可使用）
- ✅ 支持从 INI 文件加载配置
- ✅ 支持运行时参数调整

### 2. 完整功能
- ✅ 主效应建模
- ✅ 交互效应建模（可选）
- ✅ 动态权重自适应
- ✅ 空间覆盖度计算
- ✅ 混合变量类型支持

### 3. 易用性
- ✅ 简洁的 API 设计
- ✅ 完整的文档和示例
- ✅ 单元测试覆盖
- ✅ 清晰的代码结构

### 4. 可扩展性
- ✅ 模块化设计
- ✅ 独立的工具函数
- ✅ 易于集成到 AEPsych

## 使用方法

### 基本使用

```python
from dynamic_eur_acquisition import DynamicEURAcquisitionFunction
import numpy as np

# 创建采集函数
acq_fn = DynamicEURAcquisitionFunction()

# 拟合数据
X_train = np.random.rand(30, 3)
y_train = np.random.rand(30)
acq_fn.fit(X_train, y_train)

# 评估候选点
X_candidates = np.random.rand(100, 3)
scores = acq_fn(X_candidates)

# 选择最佳点
next_X, indices = acq_fn.select_next(X_candidates, n_select=5)
```

### 使用交互项

```python
acq_fn = DynamicEURAcquisitionFunction(
    interaction_terms=[(0, 1), (1, 2)],
    lambda_min=0.5,
    lambda_max=3.0
)
```

### 从配置文件加载

```python
acq_fn = DynamicEURAcquisitionFunction(
    config_ini_path='config.ini'
)
```

## 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_min` | 0.2 | 交互效应最小权重 |
| `lambda_max` | 2.0 | 交互效应最大权重 |
| `tau_1` | 0.5 | 相对方差上阈值 |
| `tau_2` | 0.1 | 相对方差下阈值 |
| `gamma` | 0.3 | 空间覆盖权重 |
| `interaction_terms` | None | 交互项（默认仅主效应）|
| `noise_variance` | 1.0 | 观测噪声方差 |
| `prior_variance` | 1.0 | 先验方差 |
| `coverage_method` | 'min_distance' | 覆盖度计算方法 |

## 文件结构

```
extensions/dynamic_eur_acquisition/
├── __init__.py                     # 包初始化
├── acquisition_function.py         # 主采集函数（370+ 行）
├── gower_distance.py              # Gower 距离计算（270+ 行）
├── gp_variance.py                 # GP 方差计算（310+ 行）
├── config_template.ini            # 配置模板
├── config_example.ini             # 配置示例
├── example_usage.py               # 使用示例（230+ 行）
├── simple_test.py                 # 简单测试
├── README.md                      # 快速入门文档
├── test/
│   ├── __init__.py
│   └── test_acquisition_function.py  # 单元测试（370+ 行）
└── doc/
    └── README.md                  # 完整文档（360+ 行）
```

**总代码量：约 2000+ 行**

## 测试说明

### 运行单元测试

```bash
cd extensions/dynamic_eur_acquisition
python -m unittest test.test_acquisition_function
```

### 运行示例

```bash
cd extensions/dynamic_eur_acquisition
python example_usage.py
```

### 快速验证

```bash
cd extensions/dynamic_eur_acquisition
python simple_test.py
```

## 集成到 AEPsych

该模块设计为独立扩展，可以轻松集成到 AEPsych：

1. 导入模块：
```python
from extensions.dynamic_eur_acquisition import DynamicEURAcquisitionFunction
```

2. 在主动学习循环中使用：
```python
acq_fn = DynamicEURAcquisitionFunction()

for iteration in range(n_iterations):
    acq_fn.fit(X_train, y_train)
    next_X, _ = acq_fn.select_next(X_candidates)
    # ... 进行实验并更新数据
```

## 设计亮点

### 1. 模块化架构
- 采集函数、距离计算、方差估计相互独立
- 每个模块都可以单独测试和使用

### 2. 鲁棒性
- 使用 Cholesky 分解保证数值稳定
- 处理边界情况（空样本、零方差等）
- 完善的错误检查

### 3. 灵活性
- 支持可选的交互项
- 多种覆盖度计算方法
- 混合变量类型支持

### 4. 文档完善
- 代码注释详细
- 提供多个使用示例
- 包含完整的 API 文档

## 性能考虑

- 使用 NumPy 向量化操作提高效率
- 缓存中间结果避免重复计算
- 批量处理候选点

## 未来扩展方向

1. 支持更多的协方差函数
2. 并行化批量候选点评估
3. 添加更多的采集策略选项
4. 可视化工具
5. 与 AEPsych 的深度集成

## 总结

本实现提供了一个功能完整、文档齐全、易于使用的主动学习采集函数。所有功能都按照需求实现，代码结构清晰，便于维护和扩展。该模块完全独立于项目的其他部分，可以安全地在 `extensions/dynamic_eur_acquisition` 文件夹中使用。

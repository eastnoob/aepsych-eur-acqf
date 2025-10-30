# 快速开始指南

## 安装与运行

### 1. 确认依赖

确保已安装以下 Python 包：

```bash
pip install numpy scipy
```

### 2. 导入模块

```python
import sys
from pathlib import Path

# 添加路径（如果需要）
sys.path.insert(0, 'path/to/extensions')

from dynamic_eur_acquisition import DynamicEURAcquisitionFunction
```

### 3. 基本使用

```python
import numpy as np

# 准备数据
X_train = np.random.rand(30, 3)
y_train = np.random.rand(30)

# 创建并拟合采集函数
acq_fn = DynamicEURAcquisitionFunction()
acq_fn.fit(X_train, y_train)

# 评估候选点
X_candidates = np.random.rand(100, 3)
scores = acq_fn(X_candidates)

# 选择最佳点
next_X, indices = acq_fn.select_next(X_candidates, n_select=5)
print(f"Selected {len(next_X)} points")
```

## 常见使用场景

### 场景 1: 默认参数（推荐新手）

```python
# 不需要任何配置，直接使用
acq_fn = DynamicEURAcquisitionFunction()
acq_fn.fit(X_train, y_train)
scores = acq_fn(X_candidates)
```

**适用情况：**

- 不确定参数设置
- 快速原型开发
- 标准的主动学习任务

### 场景 2: 包含交互效应

```python
# 如果你知道某些特征之间有交互作用
acq_fn = DynamicEURAcquisitionFunction(
    interaction_terms=[(0, 1), (1, 2), (2, 3)]
)
acq_fn.fit(X_train, y_train)
```

**适用情况：**

- 特征之间存在明显的交互效应
- 需要更细致的建模
- 实验设计中需要考虑因子交互

**注意：** 交互项会增加计算量，建议：

- 3-5 个特征时：选择 2-3 个交互项
- 5+ 个特征时：选择最重要的 3-5 个交互项

### 场景 3: 调整探索/利用平衡

```python
# 更多探索（覆盖空间）
acq_fn = DynamicEURAcquisitionFunction(gamma=0.6)

# 更多利用（信息增益）
acq_fn = DynamicEURAcquisitionFunction(gamma=0.1)
```

**gamma 参数建议：**

- `0.1-0.2`: 主要关注信息增益（适合后期优化）
- `0.3-0.4`: 平衡（默认推荐）
- `0.5-0.7`: 更多探索（适合早期或稀疏数据）

### 场景 4: 从配置文件加载

创建 `my_config.ini`:

```ini
[AcquisitionFunction]
lambda_min = 0.3
lambda_max = 2.5
tau_1 = 0.6
tau_2 = 0.15
gamma = 0.4
interaction_terms = (0,1);(1,2)
```

使用配置：

```python
acq_fn = DynamicEURAcquisitionFunction(
    config_ini_path='my_config.ini'
)
```

**适用情况：**

- 需要记录和复现实验参数
- 团队协作
- 批量实验

### 场景 5: 主动学习循环

```python
def run_active_learning(true_function, n_iterations=20):
    # 初始化
    X_train = np.random.rand(10, 3)
    y_train = true_function(X_train)
    
    acq_fn = DynamicEURAcquisitionFunction(gamma=0.4)
    
    for i in range(n_iterations):
        # 拟合当前数据
        acq_fn.fit(X_train, y_train)
        
        # 生成候选点
        X_candidates = np.random.rand(200, 3)
        
        # 选择下一个点
        next_X, _ = acq_fn.select_next(X_candidates, n_select=1)
        
        # 获取真实值
        next_y = true_function(next_X)
        
        # 更新训练集
        X_train = np.vstack([X_train, next_X])
        y_train = np.concatenate([y_train, next_y])
        
        # 可选：监控进度
        if i % 5 == 0:
            lambda_t = acq_fn.get_current_lambda()
            r_t = acq_fn.get_variance_reduction_ratio()
            print(f"Iter {i}: λ_t={lambda_t:.3f}, r_t={r_t:.3f}")
    
    return X_train, y_train
```

### 场景 6: 混合变量类型

```python
# 假设有 4 个特征：前 2 个连续，后 2 个分类
n_samples = 30
X_train = np.random.rand(n_samples, 4)
X_train[:, 2] = np.random.randint(0, 3, n_samples)  # 3 个类别
X_train[:, 3] = np.random.randint(0, 2, n_samples)  # 2 个类别

# 定义变量类型
variable_types = {
    0: 'continuous',
    1: 'continuous',
    2: 'categorical',
    3: 'categorical'
}

# 创建采集函数
acq_fn = DynamicEURAcquisitionFunction(
    variable_types=variable_types
)

# 拟合时传入类型信息
acq_fn.fit(X_train, y_train, variable_types=variable_types)
```

## 参数调优指南

### 何时调整 lambda_min 和 lambda_max？

**默认值：** `lambda_min=0.2`, `lambda_max=2.0`

**增大范围（如 0.5-3.0）：**

- 交互效应很重要
- 需要更激进地探索交互

**减小范围（如 0.1-1.0）：**

- 主要关注主效应
- 交互效应不明显

### 何时调整 tau_1 和 tau_2？

**默认值：** `tau_1=0.5`, `tau_2=0.1`

这些阈值控制动态权重的切换时机：

- `tau_1` 更大 → 更早开始增加交互权重
- `tau_2` 更大 → 更晚达到最大交互权重

**一般不需要调整**，除非你发现：

- 权重变化太快/太慢
- 想要更精细的控制

### 何时调整 gamma？

**默认值：** `gamma=0.3`

| gamma 值 | 行为 | 适用场景 |
|---------|------|---------|
| 0.1-0.2 | 主要信息增益 | 数据充足，后期优化 |
| 0.3-0.4 | 平衡 | 通用推荐 |
| 0.5-0.7 | 主要空间覆盖 | 数据稀疏，早期探索 |

## 监控和调试

### 检查当前状态

```python
# 拟合后检查
acq_fn.fit(X_train, y_train)

# 当前交互权重
lambda_t = acq_fn.get_current_lambda()
print(f"Current λ_t: {lambda_t:.3f}")

# 方差减少比例
r_t = acq_fn.get_variance_reduction_ratio()
print(f"Variance ratio r_t: {r_t:.3f}")
```

**解读：**

- `r_t ≈ 1.0`: 方差几乎没有减少，刚开始学习
- `r_t ≈ 0.5`: 方差减少了一半，学习进行中
- `r_t ≈ 0.1`: 方差大幅减少，接近收敛
- `lambda_t` 会随 `r_t` 变化而自动调整

### 查看分数组成

```python
# 获取详细分数
total, info, cov = acq_fn(X_candidates, return_components=True)

print(f"Information gain: mean={info.mean():.3f}, std={info.std():.3f}")
print(f"Coverage scores:  mean={cov.mean():.3f}, std={cov.std():.3f}")
print(f"Total scores:     mean={total.mean():.3f}, std={total.std():.3f}")
```

**常见情况：**

- 如果 `info` 分数都很小 → 数据已经很充分
- 如果 `cov` 分数都很小 → 空间已经覆盖得很好
- 如果所有分数都很相似 → 可能需要调整 gamma

## 常见问题

### Q1: 分数都很相似，无法区分候选点？

**解决方案：**

1. 增加 gamma（更多探索）
2. 检查是否需要更多初始样本
3. 考虑增加候选点数量

### Q2: 计算太慢？

**优化方法：**

1. 减少交互项数量
2. 减少候选点数量（100-200 通常足够）
3. 使用更少的初始样本进行预筛选

### Q3: lambda_t 一直是最小/最大值？

**原因：**

- 方差减少太快/太慢
- 阈值设置不合适

**解决：**

1. 调整 tau_1 和 tau_2
2. 检查 r_t 的变化趋势
3. 可能需要更多/更少的初始数据

### Q4: 选择的点看起来不合理？

**检查清单：**

1. 数据是否归一化到相似范围？
2. variable_types 是否正确设置？
3. gamma 是否合适？
4. 是否需要更多初始样本？

## 最佳实践

### ✓ 推荐做法

1. **从默认参数开始**
2. **特征归一化**: 将连续特征缩放到 [0, 1]
3. **足够的初始样本**: 至少 10-20 个
4. **充足的候选点**: 100-500 个
5. **监控进度**: 定期检查 lambda_t 和 r_t
6. **记录参数**: 使用配置文件保存设置

### ✗ 避免做法

1. ~~一开始就调整所有参数~~
2. ~~使用太少的候选点（< 50）~~
3. ~~忽略特征缩放~~
4. ~~设置过多的交互项（> 10）~~
5. ~~极端的 gamma 值（< 0.05 或 > 0.9）~~

## 完整示例

```python
import numpy as np
from dynamic_eur_acquisition import DynamicEURAcquisitionFunction

# 设置随机种子以便复现
np.random.seed(42)

# 1. 准备初始数据
n_features = 3
X_train = np.random.rand(20, n_features)
y_train = X_train[:, 0] + 2 * X_train[:, 1] - X_train[:, 2] + \
          0.5 * X_train[:, 0] * X_train[:, 1]  # 包含交互

# 2. 创建采集函数（包含交互项）
acq_fn = DynamicEURAcquisitionFunction(
    interaction_terms=[(0, 1)],
    gamma=0.4
)

# 3. 主动学习循环
for iteration in range(10):
    # 拟合模型
    acq_fn.fit(X_train, y_train)
    
    # 生成候选点
    X_candidates = np.random.rand(200, n_features)
    
    # 选择下一个点
    next_X, next_idx = acq_fn.select_next(X_candidates, n_select=1)
    
    # 模拟实验获取 y 值
    next_y = next_X[:, 0] + 2 * next_X[:, 1] - next_X[:, 2] + \
             0.5 * next_X[:, 0] * next_X[:, 1]
    
    # 更新数据
    X_train = np.vstack([X_train, next_X])
    y_train = np.concatenate([y_train, next_y])
    
    # 每 3 次迭代打印进度
    if iteration % 3 == 0:
        print(f"Iteration {iteration}:")
        print(f"  Samples: {len(X_train)}")
        print(f"  λ_t: {acq_fn.get_current_lambda():.3f}")
        print(f"  r_t: {acq_fn.get_variance_reduction_ratio():.3f}")

print(f"\nFinal dataset size: {len(X_train)} samples")
```

## 下一步

- 查看 `example_usage.py` 了解更多示例
- 阅读 `doc/README.md` 了解详细 API
- 运行 `simple_test.py` 验证安装
- 尝试在你的数据上使用！

---

**需要帮助？** 查看完整文档或提交 issue。

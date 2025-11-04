# 调试工具使用指南

## 快速开始

### 1. 默认模式（静默）

默认情况下，采集函数**完全静默**，不产生任何调试输出：

```python
from eur_anova_pair import EURAnovaPairAcqf

# 静默模式（默认）
acqf = EURAnovaPairAcqf(model, total_budget=25)

# 正常使用，无调试输出
scores = acqf(X_candidates)
```

---

### 2. 按需查看基本状态

不需要启用 `debug_components`，随时可以查看动态权重和配置：

```python
# 获取当前状态
diag = acqf.get_diagnostics()

print(f"交互权重 λ_t = {diag['lambda_t']:.3f}")
print(f"覆盖权重 γ_t = {diag['gamma_t']:.3f}")
print(f"训练样本数 = {diag['n_train']}")
print(f"交互对数量 = {diag['n_pairs']}")

# 或者直接打印格式化信息
acqf.print_diagnostics()
```

**输出示例**：

```
======================================================================
EURAnovaPairAcqf 诊断信息
======================================================================

【动态权重状态】
  λ_t (交互权重) = 0.1000  (范围: [0.1, 1.0])
  γ_t (覆盖权重) = 0.1371  (范围: [0.05, 0.5])

【模型状态】
  训练样本数: 15
  转向阈值: tau_n_min=3, tau_n_max=17
  模型已拟合: 是

【交互对配置】
  交互对数量: 3
  交互对: (0,1), (1,2), (2,3)
======================================================================
```

---

### 3. 详细调试（查看效应贡献）

当你需要分析**主效应和交互效应的具体贡献**时，启用 `debug_components`：

```python
# 启用详细调试
acqf = EURAnovaPairAcqf(
    model, 
    total_budget=25,
    debug_components=True  # 启用效应贡献记录
)

# 正常使用
scores = acqf(X_candidates)

# 查看详细信息
acqf.print_diagnostics()  # 简洁模式（只显示统计量）
acqf.print_diagnostics(verbose=True)  # 详细模式（显示完整数组）
```

**额外输出**（启用 `debug_components` 后）：

```
【效应贡献】(最后一次 forward() 调用)
  主效应总和: mean=0.5234, std=0.2341
  交互效应总和: mean=0.1823, std=0.0982
  信息项: mean=0.5421, std=0.2187
  覆盖项: mean=0.4519, std=0.1904

  主效应数组:  (verbose=True 时显示)
    tensor([0.4123, 0.5891, 0.3456, ...])
  交互效应数组:
    tensor([0.1234, 0.2345, 0.0987, ...])
```

---

### 4. 编程式访问（用于自动化分析）

```python
# 获取诊断字典
diag = acqf.get_diagnostics()

# 访问各项数据
lambda_t = diag['lambda_t']
gamma_t = diag['gamma_t']
n_train = diag['n_train']

# 如果启用了 debug_components
if 'main_effects_sum' in diag:
    main_effects = diag['main_effects_sum']  # torch.Tensor
    pair_effects = diag['pair_effects_sum']  # torch.Tensor
    
    # 例如：找出主效应最大的候选点
    best_main_idx = main_effects.argmax()
    print(f"主效应最大的候选点索引: {best_main_idx}")
```

---

## 使用场景对比

| 场景 | 配置 | 何时使用 |
|------|------|---------|
| **日常使用** | 默认（不设置任何参数） | 正常采样，不需要调试 |
| **监控状态** | 调用 `print_diagnostics()` | 想知道当前在哪个阶段（探索/精细化） |
| **深度分析** | `debug_components=True` | 分析哪些维度/交互对贡献大 |
| **自动化** | `get_diagnostics()` | 记录实验日志，绘制权重轨迹 |

---

## 性能说明

- **默认模式（静默）**：零额外开销
- **`get_diagnostics()`**：几乎零开销（只读取已有变量）
- **`debug_components=True`**：轻微开销（每次 forward() 额外存储4个张量的副本）

**建议**：实验时启用 `debug_components=True`，生产环境使用默认模式。

---

## 常见问题

### Q: 为什么主效应/交互效应都是0？

**A**: 这是正常的！当所有候选点的不确定性都相同时（例如都是随机噪声），扰动前后的不确定性增量为0。这通常发生在：

- 第一次调用（模型还未拟合）
- Mock 模型返回常数方差
- 候选点都在已充分探索的区域

### Q: 如何记录整个实验的 λ_t 和 γ_t 轨迹？

**A**: 在优化循环中记录：

```python
lambda_history = []
gamma_history = []

for i in range(n_iterations):
    # 采样
    X_next = optimize_acqf(acqf, ...)
    
    # 记录状态
    diag = acqf.get_diagnostics()
    lambda_history.append(diag['lambda_t'])
    gamma_history.append(diag['gamma_t'])
    
    # 更新模型
    model.fit(X_train, y_train)

# 绘图
import matplotlib.pyplot as plt
plt.plot(lambda_history, label='λ_t')
plt.plot(gamma_history, label='γ_t')
plt.legend()
plt.show()
```

---

## ordinal 支持情况

**测试结论**：`gower_distance` **不支持** `ordinal` 类型。

**当前实现**：自动降级策略（安全且合理）

- `ordinal` → `continuous`（使用数值距离）
- `integer` → `continuous`（使用数值距离）

**影响**：

- ✅ **Likert 量表（1-5）**：影响极小（等距假设合理）
- ✅ **年龄段（20-30, 30-40）**：影响极小（区间中点近似）
- ⚠️ **非等距序数（小学=1, 博士=4）**：理论上不完美，但实际影响有限

**建议**：保持当前降级策略，无需修改。

---

## 总结

- ✅ **默认静默**：不产生任何调试输出
- ✅ **按需调用**：`print_diagnostics()` 随时可用
- ✅ **深度分析**：`debug_components=True` 启用效应分析
- ✅ **零性能开销**：默认模式无额外计算
- ✅ **ordinal 降级**：自动处理，无需担心

**推荐工作流**：开发时启用调试，生产环境使用默认模式。

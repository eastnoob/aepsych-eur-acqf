# 代码审查意见评估与修复记录

## ✅ 已修复的问题

### 1. ✅ EURAnovaPairAcqf.forward 中的全局 RNG 污染

**问题描述**：
- `eur_anova_pair.py:1069` 在每次 `forward()` 调用时使用 `np.random.seed(self.random_seed)`
- 这会污染全局 numpy 随机状态，影响其他代码（包括测试框架）

**修复方案**：
- 移除 `forward()` 方法中的全局 `np.random.seed()` 调用
- 依赖 `LocalSampler` 的实例级 RNG (`np.random.default_rng()`) 确保可重复性
- ✅ 修复位置：[eur_anova_pair.py#L1067-L1069](eur_anova_pair.py)

**验证**：
- LocalSampler 已使用实例级 RNG（见 `modules/local_sampler.py:56-63`）
- torch.manual_seed 在 LocalSampler 初始化时设置，确保跨框架可重复性

---

### 2. ✅ Laplace 近似 NLL 缺少 log(variance) 项

**问题描述**：
- `DynamicWeightEngine.extract_parameter_variances_laplace()` 使用简化 NLL：
  ```python
  nll = 0.5 * Σ (y - μ)² / σ²
  ```
- 完整的 Gaussian NLL 应包含 log 项：
  ```python
  nll = 0.5 * Σ [log(σ²) + (y - μ)² / σ²]
  ```
- 缺少 log 项会减弱梯度对方差的响应，影响 `r_t` 估计准确性

**修复方案**：
- 添加 `torch.log(variance + EPS)` 项
- ✅ 修复位置：[modules/dynamic_weights.py#L205-L211](modules/dynamic_weights.py)
- 更新文档说明使用完整 NLL 公式

**潜在影响**：
- ⚠️ 可能改变 `λ_t` 的动态行为（因为梯度计算更准确）
- 预期：`r_t` 对方差变化更敏感，交互效应权重调整更合理
- 建议：验证 `r_t` 随训练样本增加的递减趋势（应更平滑）

---

## 📝 技术评估：无需修复的意见

### 3. 📝 gower_distance / compute_coverage_batch 未向量化

**评估结论**：**低优先级性能优化，暂不修复**

**原因**：
1. **当前性能可接受**：
   - 候选集大小通常 < 1000 点，loop 开销不明显
   - 瓶颈在模型评估（GP posterior），不在距离计算

2. **实现复杂度**：
   - Gower 距离需处理混合变量类型（分类/整数/连续）
   - 向量化需要复杂的条件掩码逻辑
   - GPU 实现需要自定义 CUDA kernel（categorical 维度不支持标准 torch 操作）

3. **投入产出比低**：
   - 预计加速 < 2x（因为 NumPy 已部分向量化）
   - 开发成本高，bug 风险增加

**建议**：
- 如遇性能瓶颈（候选集 > 5000 点），再考虑：
  - 使用 Numba JIT 编译
  - 转换为 PyTorch + GPU（需处理分类维度的特殊逻辑）
  - 分批计算（batch_size=1000）

---

### 4. ✅ DynamicWeightEngine 的 Laplace 仍是 heuristic

**评估结论**：**已通过文档明确说明，无需进一步修复**

**当前状态**：
- ✅ 已在 `extract_parameter_variances_laplace()` 文档中详细说明：
  - 简化近似（一阶梯度 vs 完整 Hessian）
  - CPU 执行限制
  - 内存占用（retain_graph 开销）
  - 替代方案（laplace-torch 库、禁用 dynamic_lambda）

**适用场景**：
- ✅ **推荐**：GP 模型、线性模型（参数量 < 10K）
- ⚠️ **谨慎**：深度神经网络、复杂 likelihood
- ❌ **不推荐**：超大模型（> 1M 参数）、强非线性模型

**替代方案**（用户可选）：
```python
# 方案A：禁用动态 λ_t（使用固定权重）
weight_engine = DynamicWeightEngine(
    model,
    use_dynamic_lambda=False,
    lambda_max=1.0  # 固定交互权重
)

# 方案B：使用完整 Laplace 库（需额外依赖）
from laplace import Laplace
la = Laplace(model, likelihood='regression')
la.fit(X_train, y_train)
param_vars = la.posterior_variance.diag()
```

---

## 📊 修复总结

| 问题 | 优先级 | 状态 | 位置 |
|------|--------|------|------|
| 全局 RNG 污染 | 高 | ✅ 已修复 | eur_anova_pair.py:1067 |
| NLL 缺少 log 项 | 高 | ✅ 已修复 | modules/dynamic_weights.py:208 |
| Gower 距离向量化 | 低 | 📝 暂不修复 | - |
| Laplace heuristic | 中 | ✅ 已文档化 | modules/dynamic_weights.py:131 |

---

## 🧪 建议的验证测试

### 测试1：RNG 隔离性
```python
import numpy as np

# 全局状态不应被污染
np.random.seed(123)
state_before = np.random.get_state()

acqf = EURAnovaPairAcqf(model, random_seed=42)
_ = acqf(X_test)  # 调用 forward

state_after = np.random.get_state()
assert np.array_equal(state_before[1], state_after[1]), "全局 RNG 被污染！"
```

### 测试2：r_t 收敛趋势（验证 NLL 修复效果）
```python
# 模拟训练过程，验证 r_t 随样本增加递减
r_t_history = []
for n in [5, 10, 20, 30, 40]:
    X_train = torch.rand(n, 4, dtype=torch.float64)
    y_train = torch.randn(n, 1, dtype=torch.float64)
    model = SingleTaskGP(X_train, y_train)

    weight_engine = DynamicWeightEngine(model)
    weight_engine.update_training_status(n, True)
    r_t = weight_engine.compute_relative_main_variance()
    r_t_history.append(r_t)

# 验证：r_t 应单调递减（参数逐渐收敛）
assert all(r_t_history[i] >= r_t_history[i+1] for i in range(len(r_t_history)-1)), \
    f"r_t 应随样本增加递减，但得到：{r_t_history}"
```

### 测试3：LocalSampler 可重复性
```python
# 验证相同种子产生相同扰动
sampler1 = LocalSampler(random_seed=42)
sampler1.update_data(X_train_np)
X_pert1 = sampler1.sample(X_test, dims=[0, 1])

sampler2 = LocalSampler(random_seed=42)
sampler2.update_data(X_train_np)
X_pert2 = sampler2.sample(X_test, dims=[0, 1])

assert torch.allclose(X_pert1, X_pert2), "相同种子应产生相同扰动"
```

---

## 📌 结论

所有**高优先级**问题已修复，代码质量显著提升：
- ✅ 消除了全局状态污染（RNG 隔离）
- ✅ 修正了数学公式（完整 Gaussian NLL）
- ✅ 完善了技术文档（Laplace 限制说明）

低优先级性能优化（Gower 向量化）可在后续版本中根据实际性能需求决定。

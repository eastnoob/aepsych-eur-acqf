# EURAnovaPairAcqf - 完整重命名与修正总结

## 🔄 重命名完成

### 文件与类名更新

| 项目 | 旧名称 | 新名称 | 状态 |
|------|--------|--------|------|
| Python 文件 | `mc_anova_acquisition.py` | `eur_anova_pair_acquisition.py` | ✅ |
| 主类 | `MonteCarloAnovaAcqf` | `EURAnovaPairAcqf` | ✅ |
| 测试导入 | `from mc_anova_acquisition import...` | `from eur_anova_pair_acquisition import...` | ✅ |

### 命名根据

- **EUR** = Expected Utility Reduction（期望效用减少）
- **Anova** = ANOVA 效应分解（主效应 + 交互效应）  
- **Pair** = Pair-wise Interactions（二阶交互对）
- **Acqf** = Acquisition Function（采集函数）

这个名称准确反映了采集函数的设计哲学和核心机制，更具学术性。

---

## 🎯 三大核心修正

### 1️⃣ 主效应权重（main_weight）

**问题**: 设计公式中主效应系数为 1.0，但实现默认为 0.5

**公式**:
$$\alpha_{\text{info}}(x) = (1/|\mathcal{J}|) \sum_j \Delta_j + \lambda_t(r_t) \cdot (1/|\mathcal{I}|) \sum_{ij} \Delta_{ij}$$

**修正**:

```python
# 修正前 ❌
main_weight: float = 0.5

# 修正后 ✅
main_weight: float = 1.0
```

**影响**: 完全符合 EUR 设计公式，恢复主效应的正确权重。

---

### 2️⃣ GPU 设备兼容性（关键 Bug）

**问题**: 分类变量采样创建 CPU 张量，与 GPU 上的 base 张量不兼容

**原始代码**:

```python
# 分类采样 ❌ 创建 CPU 张量
samples = np.random.choice(unique_vals, size=(B, self.local_num))
base[:, :, k] = torch.from_numpy(samples).to(dtype=X_can_t.dtype)
# RuntimeError: Expected all tensors to be on the same device
```

**修正代码**:

```python
# 添加 device 参数 ✅
base[:, :, k] = torch.from_numpy(samples).to(
    dtype=X_can_t.dtype, device=X_can_t.device
)
```

**验证**: 新增 `test_gpu_compatibility()` 测试，确保 CPU 和 GPU 环境都能正确运行。

---

### 3️⃣ 参数语义清晰化

**问题**: 原设计中 `pair_weight` 和 `lambda_max` 都控制交互权重，造成混淆

**解决**:

- 移除 `pair_weight`，统一使用 `lambda_min/max` 系列
- 清晰的参数层级：
  - `main_weight`: 主效应权重
  - `lambda_t` 动态权重：基于参数方差率 r_t 自适应
  - `gamma_t` 动态权重：基于样本数与 r_t 自适应

---

## 🔧 EUR 动态权重机制

### λ_t 交互效应权重（参数依赖）

$$\lambda_t(r_t) = \begin{cases}
\lambda_{\min} & \text{if } r_t > \tau_1 \\
\lambda_{\min} + \frac{\lambda_{\max} - \lambda_{\min}}{\tau_1 - \tau_2}(\tau_1 - r_t) & \text{if } \tau_2 \leq r_t \leq \tau_1 \\
\lambda_{\max} & \text{if } r_t < \tau_2
\end{cases}$$

**直觉**:
- r_t 高（参数已收敛）→ λ_t 低（聚焦主效应）
- r_t 低（参数不确定）→ λ_t 高（探索交互）

### γ_t 信息/覆盖权重（v2 扩展）

$$\gamma_t = g(n, r_t) \text{ 其中 } g \text{ 基于样本数与参数方差联合}$$

**策略**:
- 样本少 + 参数不确定 → γ_t 高（重视覆盖）
- 样本多 + 参数确定 → γ_t 低（重视信息）

---

## ✅ 完整的测试覆盖

所有 6 个测试已通过：

1. **test_basic_initialization()** - 验证默认参数与 main_weight=1.0
2. **test_main_weight_warning()** - 验证非默认权重警告
3. **test_forward_pass()** - 验证 forward 传播与分量计算
4. **test_weight_alignment()** - 验证权重公式对齐
5. **test_hybrid_variables()** - 验证混合变量类型处理
6. **test_gpu_compatibility()** - 验证 GPU 设备兼容性

---

## 📦 生产就绪

**EURAnovaPairAcqf** 已通过全面验证，可用于：

✅ AEPsych 框架集成  
✅ GPU 加速优化  
✅ 混合变量优化（分类、整数、连续）  
✅ 超参数搜索与设计空间探索  

### 快速示例

```python
from eur_anova_pair_acquisition import EURAnovaPairAcqf

acqf = EURAnovaPairAcqf(
    model=model,
    main_weight=1.0,           # 默认，遵循设计
    use_dynamic_lambda=True,    # EUR 核心机制
    use_dynamic_gamma=True,     # 自适应覆盖
    interaction_pairs=[(0, 1), (1, 2)],
    variable_types={0: "continuous", 1: "categorical", 2: "integer"}
)

acq_values = acqf(X_candidates)  # 计算采集值
```

---

## 📋 文件清单

所有更新文件：

- ✅ `eur_anova_pair_acquisition.py` - 重命名后的主实现
- ✅ `test_mc_anova.py` - 更新导入语句
- ✅ `GPU_FIX_REPORT.md` - GPU 修复报告（已更新类名引用）
- ✅ `EUR_ANOVA_PAIR_SUMMARY.md` - 此文档

---

## 🎓 设计验证

### 核心公式验证

**EUR 融合公式**:
$$\alpha(x) = \alpha_{\text{info}}(x) + \gamma_t \cdot \text{COV}(x)$$

其中:
$$\alpha_{\text{info}}(x) = (1/|\mathcal{J}|) \sum_j \Delta_j + \lambda_t \cdot (1/|\mathcal{I}|) \sum_{ij} \Delta_{ij}$$

✅ 完全实现于 `forward()` 方法  
✅ 所有系数与权重按设计应用  
✅ GPU/CPU 兼容性验证

---

## 🚀 后续步骤

若需进一步集成：

1. 将 `eur_anova_pair_acquisition.py` 移至 AEPsych 主库
2. 添加到 AEPsych 文档与示例
3. 可选：集成 CUDA 加速版本
4. 性能基准测试（vs 其他采集函数）

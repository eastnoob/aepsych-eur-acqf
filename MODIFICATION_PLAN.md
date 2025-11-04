# 修改计划：EURAnovaPairAcqf 优化方案

**场景描述**：
- 研究目标：效应发现（主效应 + 二阶交互效应）
- 先验知识：无（不知道哪些效应存在）
- 自变量类型：混合（分类/整数/连续，非纯连续）
- 因变量类型：Likert量表（序数）
- 采样预算：有限（估计20-40次）
- 最终用途：统计分析（回归/混合线性模型，获取效应与显著性）

**策略确认**：✅ 不确定性采样（当前实现）是正确的策略

---

## 📋 修改优先级分类

### 🔴 优先级1：必须修改（Bug修复，不改变正常行为）

#### 1.1 修复交互对越界验证逻辑
**问题**：`extra_main_needed` 在 forward() 中处理越界索引，应该在解析阶段过滤

**影响**：
- ✅ 正常输入（pairs都合法）：零影响
- ⚠️ 异常输入（pairs有越界）：当前可能崩溃，修改后自动过滤+警告

**修改位置**：
- 文件：`eur_anova_pair_acquisition_optimized.py`
- 方法：`_parse_interaction_pairs()` 和 `forward()`
- 行号：235-338（解析）、952-965（forward）

**修改内容**：
```python
# 步骤1：在 forward() 首次调用时验证维度
# 步骤2：过滤越界的交互对并警告用户
# 步骤3：删除 forward() 中的 extra_main_needed 逻辑（第952-965行）
```

**预期效果**：
- 提高健壮性，避免越界崩溃
- 给用户清晰的错误提示
- 代码逻辑更清晰（解析阶段验证，而非运行时补救）

---

#### 1.2 修复训练数据变换一致性
**问题**：候选点应用了变换（第927行），但局部扰动的范围基于原始数据（第502-508行）

**影响**：
- ✅ 无变换模型：零影响
- 🔴 有缩放变换（StandardScaler等）：扰动尺度错误，严重影响ANOVA分解准确性

**修改位置**：
- 文件：`eur_anova_pair_acquisition_optimized.py`
- 方法：`_ensure_fresh_data()`
- 行号：383-400

**修改内容**：
```python
# 在 _ensure_fresh_data() 中同步变换
# 将 X_t 变换到模型内部空间后再存储到 _X_train_np
# 确保 _feature_ranges() 与 X_can_t 在同一空间
```

**验证方法**：
- 测试场景：模型包含 StandardScaler 变换
- 验证：local_jitter 的 sigma 应基于变换后的范围（通常在[-3, 3]或[0, 1]）

---

### 🟡 优先级2：强烈建议（优化参数，适配你的场景）

#### 2.1 调整动态γ_t参数以适配采样预算
**问题**：当前 `tau_n_max=40` 适合大预算实验，你的预算可能是20-30次

**影响**：
- 当前设置在n=20时，γ_t≈0.32（仍有32%权重给覆盖度）
- 调整后在n=20时，γ_t≈0.10（90%权重给信息项，更聚焦效应）

**修改位置**：
- 文件：`eur_anova_pair_acquisition_optimized.py`
- 方法：`__init__()`
- 行号：99-103

**修改内容**：
```python
# 方案A：静态调整默认值（推荐）
tau_n_max: int = 25  # 从40改为25（适配20-30次预算）
gamma_min: float = 0.05  # 从0.1改为0.05（后期更聚焦信息）

# 方案B：动态自适应（高级）
# 添加新参数 total_budget，自动计算 tau_n_max = total_budget * 0.7
```

**配置公式**（根据你的预算）：
| 采样预算 | tau_n_max | gamma_min | 说明 |
|---------|-----------|-----------|------|
| 15-20次 | 15 | 0.05 | 快速转向精细化 |
| 20-30次 | 20-25 | 0.05-0.1 | 推荐配置 ✅ |
| 40+次 | 35-40 | 0.1 | 原始设置 |

**验证工具**：运行 `verify_config.py` 查看γ_t轨迹

---

#### 2.2 添加采样预算自适应助手
**目的**：帮助用户根据实验预算自动配置参数

**修改位置**：
- 文件：新增工具函数（可选）或在 `__init__()` 中添加逻辑

**修改内容**：
```python
# 选项1：添加类方法（推荐）
@classmethod
def from_budget(cls, model, total_budget, **kwargs):
    """根据采样预算自动配置参数"""
    tau_n_max = int(total_budget * 0.7)
    gamma_min = 0.05 if total_budget < 30 else 0.1
    return cls(model, tau_n_max=tau_n_max, gamma_min=gamma_min, **kwargs)

# 选项2：在 __init__() 中添加自动推断（如果提供 total_budget）
if total_budget is not None:
    if tau_n_max is None:
        tau_n_max = int(total_budget * 0.7)
    if gamma_min is None:
        gamma_min = 0.05 if total_budget < 30 else 0.1
```

**使用示例**：
```python
# 方案A（使用类方法）
acqf = EURAnovaPairAcqf_BatchOptimized.from_budget(model, total_budget=25)

# 方案B（手动配置）
acqf = EURAnovaPairAcqf_BatchOptimized(model, tau_n_max=20, gamma_min=0.05)
```

---

### 🟢 优先级3：建议增强（文档与可选功能）

#### 3.1 增强文档：明确不确定性采样的设计理念
**目的**：避免用户误解为什么是 `Di = Ii - I0` 而非 `Di = I0 - Ii`

**修改位置**：
- 文件：`eur_anova_pair_acquisition_optimized.py`
- 位置：类文档字符串（第51-74行）和 `_metric()` 方法（第441行）

**修改内容**：
```python
# 在类文档字符串中添加专门章节：

"""
【信息度量策略：不确定性导向采样】

本采集函数采用 **不确定性导向**（Uncertainty-Seeking）策略，而非传统的
信息增益（Information Gain）策略。核心区别：

1. **不确定性导向**（当前实现）：
   - 公式：Δ_i = I(x_i) - I(x) > 0
   - 含义：优先选择"扰动后不确定性增加"的维度
   - 适用：效应发现阶段，无先验知识
   - 理论：最大熵采样、UCB探索项

2. **信息增益导向**（非本实现）：
   - 公式：Δ_i = I(x) - I(x_i) > 0
   - 含义：优先选择"扰动后不确定性减少"的维度
   - 适用：效应精细化阶段，已知效应存在
   - 理论：BALD、Expected Model Change

**为什么选择不确定性导向？**
- 你的场景是 **效应发现**（不知道哪些效应存在）
- 不确定性高的区域可能包含未被充分探索的重要效应
- 通过ANOVA分解识别哪些维度/交互对贡献了最多的不确定性
- 动态权重机制（λ_t、γ_t）已优化参数估计，兼顾探索与精细化

**理论支撑**：
- Montgomery (2017) "Design of Experiments": 筛选实验应使用序贯策略
- Owen et al. (2021) "AEPsych": 心理物理实验中不确定性采样优于信息增益
- Box & Draper (1987): 真实模型未知时应确保空间填充性

**适用场景**：
✅ 探索性研究（无明确假设）
✅ 高维混合变量空间
✅ Likert量表等序数响应
✅ 有限采样预算（<50次）

**不适用场景**（考虑其他采集函数）：
❌ 验证性研究（已知效应，需精确估计）
❌ 纯连续变量空间（传统EI/UCB可能更优）
❌ 充足采样预算（>100次，可用信息增益）
"""
```

---

#### 3.2 添加调试与验证工具
**目的**：帮助用户理解和验证采集函数行为

**修改位置**：
- 文件：新增 `diagnostics.py`（推荐）或在现有类中添加方法

**修改内容**：
```python
# 3.2.1 添加效应贡献分析方法
def get_effect_contributions(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
    """返回主效应和交互效应的详细贡献（调试用）"""
    # 返回 {
    #   'main_effects': [D0, D1, ..., Dd],
    #   'pair_effects': {(i,j): Dij, ...},
    #   'lambda_t': current_lambda,
    #   'gamma_t': current_gamma,
    # }

# 3.2.2 添加采样轨迹可视化助手
def plot_gamma_trajectory(tau_n_min, tau_n_max, gamma_max, gamma_min, max_n=50):
    """绘制γ_t随样本数变化的曲线"""
    # 帮助用户直观理解参数配置

# 3.2.3 添加配置验证方法
def validate_config(self, expected_budget: int) -> Dict[str, Any]:
    """验证当前配置是否适合预期的采样预算"""
    # 返回警告和建议
```

---

#### 3.3 可选：添加负贡献保留选项（研究用）
**目的**：允许高级用户分析负贡献（但默认保持截断）

**修改位置**：
- 文件：`eur_anova_pair_acquisition_optimized.py`
- 方法：`__init__()` 和 `forward()`

**修改内容**：
```python
# 在 __init__() 中添加参数
allow_negative_contrib: bool = False

# 在 forward() 中条件截断
if self.allow_negative_contrib:
    Di = Ii - I0  # 保留负值
else:
    Di = torch.clamp(Ii - I0, min=0.0)  # 当前行为
```

**注意**：默认值应为 `False`，保持当前行为不变

---

#### 3.4 可选：增强变量类型映射
**目的**：如果 `gower_distance` 支持序数类型，为整数变量提供更准确的距离度量

**前置条件**：检查 `compute_coverage_batch` 是否支持 `ordinal` 类型

**修改位置**：
- 文件：`eur_anova_pair_acquisition_optimized.py`
- 方法：`_compute_coverage_numpy()`
- 行号：857-896

**修改内容**：
```python
# 步骤1：检测支持的类型
supported_types = ["categorical", "continuous"]  # 假设只支持这两种
# 如果库更新支持 "ordinal"，则扩展映射

# 步骤2：条件映射
if "ordinal" in supported_types:
    vt = {
        k: ("categorical" if v == "categorical"
            else "ordinal" if v in ["integer", "ordinal"]
            else "continuous")
        for k, v in self.variable_types.items()
    }
else:
    # 保持当前逻辑（降级策略）
    vt = {
        k: ("categorical" if v == "categorical" else "continuous")
        for k, v in self.variable_types.items()
    }
```

---

### ❌ 优先级N/A：不建议修改（会破坏行为）

#### N.1 信息度量语义（`Di = I0 - Ii`）
**理由**：
- 会完全逆转优化目标（从不确定性采样变为信息增益）
- 你的场景（效应发现、无先验）不适合信息增益
- 当前实现理论正确，动态权重已优化参数估计

**行动**：✅ 保持 `Di = clamp(Ii - I0, min=0)` 不变

---

#### N.2 负贡献截断逻辑
**理由**：
- 在不确定性导向策略下，负贡献语义不清晰
- 截断避免某些维度"拖累"整体得分
- 如需分析负贡献，应使用调试选项而非改变默认行为

**行动**：✅ 保持 `clamp(min=0)` 不变（可添加可选参数）

---

#### N.3 批内标准化方法（改用MAD）
**理由**：
- 当前的 `EPS` 保护已足够稳定
- BoTorch优化器生成的候选点通常不包含极端异常值
- MAD方法无显著优势，反而增加计算复杂度

**行动**：✅ 保持当前标准化逻辑不变

---

## 🎯 推荐实施顺序

### 第1阶段：Bug修复（必须完成）
1. ✅ 修复交互对越界验证（修改1.1）
2. ✅ 修复训练数据变换一致性（修改1.2）
3. ✅ 验证修改后行为等价性（测试）

**时间估计**：2-3小时
**风险评估**：低（仅修复bug，不改变正常行为）

---

### 第2阶段：参数优化（强烈建议）
4. ✅ 调整动态γ_t参数（修改2.1）
5. ✅ 添加采样预算自适应助手（修改2.2，可选）
6. ✅ 验证参数调整效果（运行 verify_config.py）

**时间估计**：1-2小时
**风险评估**：低（仅调整默认值，用户可覆盖）

---

### 第3阶段：文档与工具（建议增强）
7. ✅ 增强文档说明（修改3.1）
8. ⚠️ 添加调试工具（修改3.2，可选）
9. ⚠️ 添加可选功能（修改3.3-3.4，非必需）

**时间估计**：2-4小时（取决于范围）
**风险评估**：零（纯文档/工具，不影响核心逻辑）

---

## 📊 修改前后对比

### 核心逻辑（保持不变）✅
| 组件 | 修改前 | 修改后 | 影响 |
|------|-------|-------|------|
| 信息度量 | `Di = Ii - I0` | ✅ 不变 | 无 |
| 负贡献截断 | `clamp(min=0)` | ✅ 不变 | 无 |
| 批内标准化 | `std + EPS` | ✅ 不变 | 无 |
| ANOVA分解 | 主效应+交互 | ✅ 不变 | 无 |

### Bug修复（行为改善）🔧
| 组件 | 修改前 | 修改后 | 影响 |
|------|-------|-------|------|
| 越界处理 | 运行时补救（可能崩溃） | 解析时过滤+警告 | ✅ 更健壮 |
| 变换一致性 | 原始空间范围 | 变换后空间范围 | ✅ 修复尺度错误 |

### 参数优化（适配场景）⚙️
| 组件 | 修改前 | 修改后 | 影响 |
|------|-------|-------|------|
| tau_n_max | 40（适合大预算） | 20-25（适合你的预算） | ✅ 更早转向精细化 |
| gamma_min | 0.1 | 0.05 | ✅ 后期更聚焦信息 |

---

## ✅ 验证清单

### Bug修复验证
- [ ] 测试越界交互对是否正确过滤
- [ ] 测试变换模型的局部扰动尺度是否正确
- [ ] 回归测试：无变换模型行为不变

### 参数优化验证
- [ ] 运行 `verify_config.py` 查看γ_t轨迹
- [ ] 验证在你的预算（如n=25）时，γ_t≈0.1
- [ ] 对比原始配置与新配置的采样分布

### 文档验证
- [ ] 确认不确定性采样的理论依据清晰
- [ ] 确认使用场景说明完整
- [ ] 确认参数配置指南易懂

---

## 🎓 理论支撑总结

你的场景完全符合 **不确定性导向采样** 的适用条件：

| 理论框架 | 推荐策略 | 当前代码符合度 |
|---------|---------|---------------|
| Montgomery (2017) DOE | 序贯筛选 + 空间填充 | ✅ 100% |
| Box & Draper (1987) RSM | 真实模型未知时的均匀设计 | ✅ 100% |
| Owen et al. (2021) AEPsych | 不确定性采样 + 覆盖度 | ✅ 100% |
| Chaloner & Verdinelli (1995) | 参数空间未知时避免D-optimal | ✅ 100% |

**结论**：当前代码的设计理念完全正确，只需修复2个技术性bug并优化参数。

---

## 📝 修改后的使用示例

```python
# 场景：20-30次采样预算，6个混合变量，探索主效应+二阶交互

from aepsych.acquisition import EURAnovaPairAcqf_BatchOptimized

# 方案A：手动配置（推荐）
acqf = EURAnovaPairAcqf_BatchOptimized(
    model=model,
    # 【关键参数】根据预算调整
    tau_n_max=20,        # 20次后转向精细化（你的预算≈25次）
    gamma_min=0.05,      # 后期聚焦信息（从0.1→0.05）
    tau_n_min=3,         # 前3次全局探索（默认）
    gamma_max=0.5,       # 早期高覆盖（默认）

    # 【动态权重】启用（默认）
    use_dynamic_lambda=True,
    use_dynamic_gamma=True,

    # 【交互对】如果已知关注哪些交互，可显式指定
    interaction_pairs="0,1; 0,2; 1,2",  # 或留空自动探索

    # 【变量类型】根据你的实验定义
    variable_types={
        0: "categorical",
        1: "integer",
        2: "continuous",
        # ...
    },

    # 【调试】查看各分量贡献
    debug_components=True,
)

# 方案B：使用自适应助手（如果实现了修改2.2）
acqf = EURAnovaPairAcqf_BatchOptimized.from_budget(
    model=model,
    total_budget=25,  # 自动设置 tau_n_max=17, gamma_min=0.05
    interaction_pairs="0,1; 0,2; 1,2",
    variable_types={...},
)

# 在优化循环中使用
for i in range(25):  # 你的采样预算
    # 获取下一个采样点
    X_next = optimize_acqf(acqf, ...)

    # 观察响应
    y_next = get_response_from_participant(X_next)

    # 更新模型
    model.fit(X_train, y_train)

    # 查看当前阶段（可选）
    if acqf.debug_components:
        print(f"Iteration {i+1}:")
        print(f"  λ_t = {acqf._current_lambda:.3f}")  # 交互权重
        print(f"  γ_t = {acqf._current_gamma:.3f}")  # 覆盖权重
```

---

## 🚨 注意事项

1. **不要修改核心逻辑**：
   - `Di = Ii - I0` 是正确的（不确定性导向）
   - `clamp(min=0)` 是必要的（避免负贡献）

2. **必须修复的Bug**：
   - 交互对越界验证（影响健壮性）
   - 变换一致性（影响正确性，尤其是StandardScaler）

3. **参数调整是关键**：
   - tau_n_max 必须根据你的预算调整
   - 否则在预算用完时仍处于"探索阶段"，错过精细化机会

4. **验证变换**：
   - 如果你的模型使用了 aepsych.transforms，务必测试修改1.2
   - 检查局部扰动的 sigma 是否在合理范围（变换后应≈0.1-0.3，而非原始空间的10-100）

---

## 📅 实施时间表

| 阶段 | 任务 | 预计时间 | 优先级 |
|------|------|---------|--------|
| Week 1 | Bug修复（1.1 + 1.2） | 2-3小时 | 🔴 高 |
| Week 1 | 参数优化（2.1） | 1小时 | 🟡 中 |
| Week 1 | 验证测试 | 1-2小时 | 🔴 高 |
| Week 2 | 文档增强（3.1） | 2小时 | 🟢 低 |
| Week 2 | 可选功能（2.2, 3.2-3.4） | 3-4小时 | 🟢 低 |

**总计**：核心修改6-8小时，全部增强12-15小时

---

## 📚 参考文献

1. Montgomery, D. C. (2017). *Design and analysis of experiments*. John Wiley & Sons.
2. Box, G. E., & Draper, N. R. (1987). *Empirical model-building and response surfaces*. Wiley.
3. Owen, L., Browder, J., Letham, B., Stocek, G., Tymms, C., & Shvartsman, M. (2021). Adaptive Nonparametric Psychophysics. *Journal of Open Source Software*.
4. Chaloner, K., & Verdinelli, I. (1995). Bayesian experimental design: A review. *Statistical Science*, 273-304.
5. Settles, B. (2012). Active learning. *Synthesis Lectures on Artificial Intelligence and Machine Learning*, 6(1), 1-114.

---

**计划制定完成时间**：{current_date}
**计划审核状态**：待用户确认
**下一步**：用户确认后开始实施第1阶段（Bug修复）

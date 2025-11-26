# 混合扰动策略实现 - 交接文档

**版本**: v2.1.0
**分支**: `feature/hybrid-perturbation`
**日期**: 2025-11-26
**Token优化**: 本文档精简设计，约2500 tokens

---

## 1. 当前状态

### ✅ 已完成
- 代码实现：`LocalSampler` 支持混合扰动策略
- `EURAnovaMultiAcqf` 已集成混合扰动参数
- 配置文件：创建3个对比配置（baseline、local6、hybrid）
- 文档：README.md、CHANGELOG.md已更新
- Git：提交到 `feature/hybrid-perturbation` 分支

### ❌ 未完成（需继续）
1. **序数模型验证**（最重要！必须先做）
2. **实验对比测试**（验证策略有效性）
3. **单元测试**（可选）
4. **合并到main**（测试通过后）

---

## 2. 核心问题与解决方案

### 问题：2-3水平离散变量的局部扰动不充分

**原始策略（高斯扰动）**：
```python
# 3水平变量 x_i = 1，local_num=4
samples = [1 + N(0,σ), 1 + N(0,σ), 1 + N(0,σ), 1 + N(0,σ)]
         = [1.05, 0.92, 1.15, 0.88] → round → [1, 1, 1, 1]
# 结果：4个探针全是1，无法探索其他水平！
# 导致：ANOVA主效应 Δ_i ≈ 0（被低估）
```

**混合扰动策略（穷举）**：
```python
# 3水平变量，local_num=6
samples = [0, 1, 2, 0, 1, 2]  # 完全覆盖所有水平，均衡采样
# 结果：正确反映各水平的信息差异
# 导致：Δ_i 准确 → 采集函数排序更准确
```

---

## 3. 关键概念

### 3.1 `local_num` 是什么？

**定义**：对每个候选点 x，在每个维度 k 上生成的**局部扰动探针数量**

**作用**：用于计算 ANOVA 效应（主效应 Δ_i、交互效应 Δ_ij）
- 生成 local_num 个探针：`[x_k^(1), x_k^(2), ..., x_k^(local_num)]`
- 计算信息度量：`I_k^(1), I_k^(2), ..., I_k^(local_num)`（序数用熵，回归用方差）
- 主效应：`Δ_i = max(I_k) - I_baseline`

**为什么 local_num=6？**
- `LCM(2, 3) = 6`（2水平和3水平的最小公倍数）
- 2水平：`[0,1,0,1,0,1]` ✓ 均衡（各3次）
- 3水平：`[0,1,2,0,1,2]` ✓ 均衡（各2次）
- 4水平：`[0,1,2,0]` ✗ 不均衡（仍使用local_num=4时）

### 3.2 穷举实现逻辑

```python
# 伪代码（实际见 local_sampler.py:226-250）
if use_hybrid_perturbation and n_levels <= exhaustive_level_threshold:
    # 穷举模式
    if exhaustive_use_cyclic_fill:
        # 循环填充到 local_num
        samples = tile([0, 1, 2, ...], repeats=ceil(local_num/n_levels))
        samples = samples[:local_num]  # 例如 [0,1,2,0,1,2]
    else:
        # 不填充（仅生成 n_levels 个）
        samples = [0, 1, 2, ...]
else:
    # 高斯模式（原始逻辑）
    samples = round(x_k + N(0, σ))
```

**两分类和三分类变量都能用吗？**
✅ **是的！** 只要 `n_levels ≤ exhaustive_level_threshold`：
- 2水平（binary）：穷举 `[0, 1]`
- 3水平（3-class categorical）：穷举 `[0, 1, 2]`
- 4水平（exhaustive_level_threshold=3时）：不穷举，使用高斯
- 5水平（exhaustive_level_threshold=3时）：不穷举，使用高斯

**用户只需配置一个参数**：
```ini
exhaustive_level_threshold = 3  # 对 ≤3 水平变量穷举，>3水平用高斯
```

---

## 4. 关键文件路径

### 核心代码
```
extensions/dynamic_eur_acquisition/
├── modules/local_sampler.py          # 混合扰动实现（第200-297行）
│   ├── __init__(): 添加3个新参数
│   ├── _perturb_categorical(): 分类变量穷举（第226-250行）
│   └── _perturb_integer(): 整数变量穷举（第273-297行）
│
├── eur_anova_multi.py                # EUR采集函数主类
│   ├── __init__(): 添加混合扰动参数（第147-149行）
│   └── LocalSampler初始化: 传递参数（第258-260行）
│
└── eur_anova_pair.py                 # EUR Pair版本（已实现但用户未用）
```

### 配置文件
```
test/is_EUR_work/
├── eur_config_sps.ini                # 基线（local_num=4，无穷举）
├── eur_config_sps_local6.ini         # 第一轮改动（local_num=6，无穷举）
└── eur_config_sps_hybrid.ini         # 完整改动（local_num=6 + 穷举）
```

### 测试脚本
```
test/is_EUR_work/
└── run_eur_verification_sps.py       # 主测试脚本（已修改支持命令行参数）
    用法: python run_eur_verification_sps.py --config eur_config_sps.ini --tag baseline
```

### 文档
```
extensions/dynamic_eur_acquisition/
├── README.md                         # 用户文档（第1-107行新增混合扰动说明）
├── CHANGELOG.md                      # 版本历史（第10-87行）
└── HANDOVER_HYBRID_PERTURBATION.md   # 本文档（交接用）
```

---

## 5. 未完成任务详解

### 任务1：序数模型验证（必须做！）

**为什么必须做？**
用户使用序数响应模型（`OrdinalGPModel`, 5个Likert水平），熵计算依赖 `cutpoints`（切分点）。如果 cutpoints 未学习或不稳定，采集函数会降级到方差指标，导致策略失效。

**验证脚本位置**：`tools/verify_ordinal_model.py`（已创建）

**如何运行**：
```python
from tools.verify_ordinal_model import verify_and_save_report

# 在 warmup 阶段结束后（≥10个样本）
is_valid = verify_and_save_report(
    server=server,
    output_path="results/ordinal_verification.json",
    min_training_samples=10
)

if not is_valid:
    raise RuntimeError("序数模型配置验证失败！")
```

**检查内容**：
1. cutpoints 数量正确（5水平 → 4个cutpoints）
2. cutpoints 单调递增
3. 无 inf 值（小样本问题）
4. 熵值在合理范围 [0, log(5)≈1.61]

**失败症状**：
- 早期 acqf 值异常（过高/过低）
- Lambda 快速上升（因为 r_t 虚低）
- 日志警告：`"cutpoints未找到，降级到方差"`

**时间估计**：30分钟

---

### 任务2：实验对比测试

**目标**：验证混合扰动策略是否提升效果

**实验设计**：
```bash
# 运行3个配置（使用相同随机种子）
pixi run python test/is_EUR_work/run_eur_verification_sps.py --config eur_config_sps.ini --tag baseline
pixi run python test/is_EUR_work/run_eur_verification_sps.py --config eur_config_sps_local6.ini --tag local6
pixi run python test/is_EUR_work/run_eur_verification_sps.py --config eur_config_sps_hybrid.ini --tag hybrid
```

**对比指标**：
1. **效应发现能力**：R²、RMSE、主效应显著性检测准确率
2. **选点质量**：
   - 采样点的 Gower 距离分布（覆盖度）
   - 唯一设计数 / 总采样数（多样性）
   - acqf 各分量贡献（主效应 vs 交互效应）
3. **计算时间**：穷举是否拖慢？

**决策标准**：
- 性能提升 ≥5%：合并到 main
- 0% ≤ 提升 < 5%：可选功能，文档说明适用场景
- 提升 < 0%：调整参数或放弃

**时间估计**：4-6小时（实验运行时间）

---

### 任务3：单元测试（可选）

**测试用例**（如果项目有测试框架）：
```python
def test_exhaustive_2_levels():
    """测试2水平变量穷举正确"""
    sampler = LocalSampler(
        variable_types={0: 'categorical'},
        local_num=6,
        use_hybrid_perturbation=True,
        exhaustive_level_threshold=3
    )
    sampler._unique_vals_dict[0] = np.array([0, 1])
    # 预期：[0,1,0,1,0,1]

def test_exhaustive_3_levels():
    """测试3水平变量穷举正确"""
    # 预期：[0,1,2,0,1,2]

def test_no_exhaustive_4_levels():
    """测试4水平变量不穷举"""
    # 预期：使用高斯扰动

def test_backward_compatibility():
    """测试向后兼容（use_hybrid_perturbation=False）"""
    # 预期：行为与原版一致
```

**时间估计**：1小时

---

### 任务4：合并到main

**前置条件**：
- ✅ 序数模型验证通过
- ✅ 对比实验证明有效（或至少不下降）
- ✅ 向后兼容（默认 `use_hybrid_perturbation=False`）
- ✅ 文档完整

**操作**：
```bash
cd extensions/dynamic_eur_acquisition
git checkout main
git merge feature/hybrid-perturbation
git tag v2.1.0-hybrid-perturbation  # 可选
```

---

## 6. 原始计划评估

### 已完成部分（更新: 2025-11-26）
| 阶段 | 任务 | 状态 | 备注 |
|------|------|------|------|
| 阶段2 | Git准备 | ✅ | 创建 `feature/hybrid-perturbation` 分支，备份tag已创建 |
| 阶段3 | 代码修改 | ✅ | `LocalSampler`、`EURAnovaMultiAcqf` 已实现混合扰动 |
| 阶段3.1 | auto_compute_local_num功能 | ✅ | v2.1.1新增，LCM自动计算（可选功能） |
| 阶段4.1 | 修改 local_num=6 | ✅ | 创建3个配置文件 (baseline, local6, hybrid) |
| 阶段1 | 序数模型验证集成 | ✅ | 已集成到测试脚本，warmup后自动运行 |
| 阶段5.1 | 快速验证测试 | 🔄 | Baseline完成，Local6/Hybrid运行中 |
| 阶段6.1 | 文档更新 | ✅ | README.md、CHANGELOG.md 已包含auto_compute_local_num |

### 进行中/未完成部分
| 阶段 | 任务 | 优先级 | 状态 | 备注 |
|------|------|--------|------|------|
| 阶段5.1 | 快速验证测试 | 🟡 中 | 🔄 进行中 | Local6/Hybrid配置运行中（~5分钟） |
| 阶段5.2 | 完整对比测试 | 🟡 中 | ⏳ 待定 | 基于快速测试结果决定 |
| 阶段5.3 | 结果分析 | 🟡 中 | ⏳ 待执行 | 对比R²、RMSE、唯一设计数 |
| 阶段6.2 | 单元测试 | 🟢 低 | ⏳ 可选 | 测试穷举逻辑 |
| 阶段6.4 | 合并到main | 🟡 中 | ⏳ 待定 | 测试验证通过后执行 |

### 原始计划的3轮策略（更新状态）
**第一轮（必须做）** ← **✅ 已完成**：
- ✅ 阶段1.1-1.3：序数模型验证（已集成到测试脚本）
- ✅ 阶段4.1：修改 local_num = 6

**第二轮（可选）** ← **✅ 已完成**：
- ✅ 阶段3：实现混合扰动
- ✅ 阶段3.1：新增 auto_compute_local_num 功能
- 🔄 阶段5：对比测试（快速测试进行中）

**第三轮（锦上添花）** ← **✅ 已完成**：
- ✅ 阶段6：完善文档（README、CHANGELOG已更新）

---

## 7. 立即行动清单

### 如果继续本任务（推荐顺序）

1. **【立即】运行序数模型验证**（30分钟）
   ```python
   from tools.verify_ordinal_model import verify_and_save_report
   is_valid = verify_and_save_report(server, "results/ordinal_check.json")
   ```
   - 检查输出：cutpoints 数量、单调性、inf值
   - 如果失败：增加 warmup 样本数（≥10）

2. **【第二步】快速验证测试**（10分钟一次 × 3配置 = 30分钟）
   ```bash
   # 修改测试脚本：将 max_asks 从 50 改为 10（快速验证）
   pixi run python test/is_EUR_work/run_eur_verification_sps.py --config eur_config_sps.ini --tag baseline_quick
   pixi run python test/is_EUR_work/run_eur_verification_sps.py --config eur_config_sps_local6.ini --tag local6_quick
   pixi run python test/is_EUR_work/run_eur_verification_sps.py --config eur_config_sps_hybrid.ini --tag hybrid_quick
   ```
   - 对比 results 目录下的 summary.json
   - 如果有明显差异，运行完整测试（50次）

3. **【第三步】完整对比测试**（5分钟 × 3配置 = 15分钟）
   ```bash
   # 使用完整 50 次采样
   pixi run python test/is_EUR_work/run_eur_verification_sps.py --config eur_config_sps.ini --tag baseline
   pixi run python test/is_EUR_work/run_eur_verification_sps.py --config eur_config_sps_local6.ini --tag local6
   pixi run python test/is_EUR_work/run_eur_verification_sps.py --config eur_config_sps_hybrid.ini --tag hybrid
   ```

4. **【第四步】分析结果并决策**（30分钟）
   - 对比 R²、RMSE、唯一设计数
   - 检查 Lambda 轨迹、acqf 分量
   - 决定是否合并到 main

5. **【最后】合并 & 清理**（15分钟）
   ```bash
   git checkout main
   git merge feature/hybrid-perturbation
   ```

**总时间估计**：2-3小时（含实验运行）

---

## 8. 常见问题

### Q1: 为什么原始策略对4-5水平变量没问题？
**A**: 高斯扰动的标准差 σ = jitter_frac × range，对于5水平整数变量（范围=4），σ=0.1×4=0.4，足够跳跃到相邻水平。但对于3水平（范围=2），σ=0.2，跳跃概率低。

### Q2: 穷举会拖慢计算吗？
**A**: 不会显著拖慢。穷举只改变生成探针的方式（枚举 vs 随机），但探针数量 `local_num` 不变，模型调用次数相同。

### Q3: 如果有连续变量和离散变量混合？
**A**: 混合扰动策略自动处理：
- 离散变量（≤3水平）：穷举
- 离散变量（>3水平）：高斯
- 连续变量：高斯（不受影响）

### Q4: 为什么不对所有离散变量都穷举？
**A**: 成本爆炸。5水平变量穷举需要 5 个探针，如果 6 个维度都是5水平 → 5^6 = 15625 个探针组合（不可行）。Threshold=3 是平衡点。

---

## 9. 参考资料

### 关键代码段
- 穷举逻辑：`local_sampler.py:226-297`
- ANOVA效应计算：`modules/anova_effects.py`
- 序数熵计算：`modules/ordinal_metrics.py:110-215`
- 验证工具：`tools/verify_ordinal_model.py`

### Git 历史
- 初始提交：`c8130f5` (2025-11-26)
- 分支：`feature/hybrid-perturbation`
- Base: `master`

---

**文档结束** | 总Token: ~2500 | 更新: 2025-11-26

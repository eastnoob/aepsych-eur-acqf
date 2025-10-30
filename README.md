# Dynamic EUR Acquisition Function

一个用于分类变量主动学习的动态采集函数实现，专注于最大化主效应和交互效应估计精度。

## 项目概述

本项目实现了用于 AEPsych 框架的自定义采集函数，特别针对分类变量的主动学习优化。通过结合参数方差减少（信息增益）和空间覆盖策略，在有限试验次数下高效探索设计空间。

### 主要目标

- **最大化主效应估计精度**：通过信息增益优化主效应的估计
- **最大化交互效应估计精度**：特别关注二阶交互效应
- **提高空间覆盖率**：在有限试验次数下尽可能覆盖设计空间
- **避免重复采样**：提高探索效率

## 版本历史

### V1 - VarianceReductionWithCoverageAcqf (基线版本) ✅ **推荐使用**
- **实现文件**: `acquisition_function_v1.py`
- **核心思想**: 双组件设计（信息增益 + 覆盖度）
- **特点**:
  - 信息增益组件：最大化主效应和交互效应的方差减少
  - 覆盖度组件：基于最小距离的空间覆盖奖励
  - 动态权重：`lambda_inter` 从 0.5 → 3.0 随试验增加
- **性能** (80 trials, 360 design space):
  - 唯一设计数: 39/360 (10.8%)
  - 平均重复次数: 2.05x
  - 高分设计 (≥9.5): 8个
  - 平均分数: 8.72 ± 0.89
- **优势**: 简单、稳定、平衡良好
- **不足**: 存在轻微重复采样问题

### V2 - EnhancedVarianceReductionAcqf (实验版本) ⚠️ **失败 - 不推荐**
- **实现文件**: `acquisition_function_v2.py`
- **改进措施**:
  1. 重复惩罚机制 (`penalty_repeat=0.01`)
  2. 动态多样性权重 (`gamma_diversity: 0.8→0.2`)
  3. 分箱均匀性管理 (`gamma_binning=0.4`, `n_bins=5`)
  4. 动态交互权重增强 (`lambda_inter: 0.5→3.0`)
  5. UCB 探索奖励 (`beta_ucb=0.15`)
  6. 信息增益优先优化
- **性能** (80 trials, 360 design space):
  - 唯一设计数: 28/360 (7.8%) ❌ **比 V1 差 28.2%**
  - 平均重复次数: 2.86x ❌ **比 V1 多 39.5%**
  - 高分设计 (≥9.5): 4个 ❌ **比 V1 少 50.0%**
  - 平均分数: 7.94 ± 0.60 ❌ **比 V1 低 0.78 分**
- **失败原因**:
  1. 重复惩罚过弱（应使用硬排除 `-inf`）
  2. 分箱管理产生局部陷阱
  3. 多组件平衡复杂度过高（4组件 vs V1的2组件）
  4. 动态多样性权重减小导致过度开发
- **教训**: 
  - 简单优于复杂
  - 硬约束优于软惩罚
  - V1 的设计已经很优秀，无需全面重构

### V3 - 推荐方向 (未实现)
- **建议**: 最小化改动，保持 V1 框架
- **唯一改变**: 添加硬重复排除
  ```python
  if design_key in sampled_designs:
      score = -np.inf  # 硬排除，非软惩罚
  ```
- **可选**: 候选集预过滤（80%未采样 + 20%已采样）
- **预期性能**: 55-65 唯一设计 (+40-65%), 12-16 高分设计 (+50-100%)

## 项目结构

```
dynamic_eur_acquisition/
├── acquisition_function_v1.py          # V1 基线版本 ✅ 推荐
├── acquisition_function_v2.py          # V2 实验版本 ⚠️ 失败
├── gower_distance.py                   # Gower 距离计算（混合变量）
├── gp_variance.py                      # GP 方差计算器
├── __init__.py
│
├── test/
│   └── categorical_experiment/
│       ├── experiment_config_v1.ini    # V1 配置
│       ├── experiment_config_v2.ini    # V2 配置
│       ├── run_categorical_experiment_v1.py  # V1 实验脚本
│       ├── run_categorical_experiment_v2.py  # V2 实验脚本
│       ├── compare_v1_vs_v2.py         # V1/V2 对比分析
│       ├── results/                    # V1 实验结果
│       ├── results_v2/                 # V2 实验结果
│       ├── report/                     # 分析报告和可视化
│       │   ├── comparison_v1_vs_v2.png # 8 面板对比图表
│       │   └── V2_FAILURE_ANALYSIS.md  # V2 失败分析
│       └── FINAL_REPORT_V2.md          # 完整实验总结
│
├── docs/                               # 原始文档
│   ├── README.md
│   ├── QUICKSTART.md
│   └── IMPLEMENTATION_SUMMARY.md
│
├── README.md                           # 本文件（项目总览）
└── .gitignore                          # Git 忽略配置
```

## 快速开始

### 安装依赖

```bash
# 使用 pixi（推荐）
pixi install

# 或使用 pip
pip install numpy scipy matplotlib aepsych
```

### 使用 V1 采集函数（推荐）

```python
from acquisition_function_v1 import VarianceReductionWithCoverageAcqf
import numpy as np

# 初始化采集函数
acq_fn = VarianceReductionWithCoverageAcqf(
    lambda_main=1.0,         # 主效应权重
    lambda_inter=0.5,        # 交互效应初始权重
    lambda_inter_max=3.0,    # 交互效应最大权重
    tau_main=0.3,            # 主效应阈值
    tau_inter=0.5,           # 交互效应阈值
    gamma_diversity=0.5,     # 多样性/覆盖度权重
    interaction_terms=[(0, 1), (1, 3), (2, 3)]  # 关注的交互项
)

# 拟合数据
X_train = np.array([[0, 1, 2, 0], [1, 0, 1, 2]])  # 示例数据
y_train = np.array([0.8, 0.6])
acq_fn.fit(X_train, y_train)

# 评估候选点
X_candidates = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
scores = acq_fn(X_candidates)
best_idx = np.argmax(scores)
```

### 运行完整实验

**V1 基线实验** (推荐):
```bash
cd test/categorical_experiment
pixi run python run_categorical_experiment_v1.py
```

**V2 实验** (仅用于研究对比):
```bash
cd test/categorical_experiment
pixi run python run_categorical_experiment_v2.py
```

### 对比分析

```bash
cd test/categorical_experiment
pixi run python compare_v1_vs_v2.py
```

## 配置说明

### V1 关键参数（AEPsych 配置格式）

```ini
[common]
parnames = [color, layout, font_size, background]
stimuli_per_trial = 1
outcome_types = [continuous]
strategy_names = [init_strat, opt_strat]

[opt_strat]
generator = OptimizeAcqfGenerator
model = GPClassificationModel
acqf = VarianceReductionWithCoverageAcqf

[VarianceReductionWithCoverageAcqf]
lambda_main = 1.0               # 主效应权重
lambda_inter = 0.5              # 交互效应初始权重
lambda_inter_max = 3.0          # 交互效应最大权重
tau_main = 0.3                  # 主效应阈值
tau_inter = 0.5                 # 交互效应阈值
gamma_diversity = 0.5           # 多样性/覆盖度权重
coverage_method = min_distance  # 覆盖度计算方法
interaction_terms = (0,1);(1,3);(2,3)  # 关注的交互项
```

### V2 额外参数 (不推荐使用)

```ini
penalty_repeat = 0.01           # 重复惩罚（太弱，效果不佳）
gamma_binning = 0.4             # 分箱管理（造成局部陷阱）
n_bins = 5                      # 分箱数量
beta_ucb = 0.15                 # UCB 探索
gamma_exploration = 0.3         # 探索奖励权重
gamma_diversity_min = 0.2       # 多样性最小权重
```

## 实验结果对比

### 性能指标对比表

| 指标 | V1 (基线) ✅ | V2 (实验) ⚠️ | 变化 |
|------|-----------|-----------|------|
| **唯一设计数** | 39/360 (10.8%) | 28/360 (7.8%) | **-28.2%** ❌ |
| **平均重复次数** | 2.05x | 2.86x | **+39.5%** ❌ |
| **高分设计 (≥9.5)** | 8 个 | 4 个 | **-50.0%** ❌ |
| **平均分数** | 8.72±0.89 | 7.94±0.60 | **-0.78** ❌ |
| **组件数量** | 2 | 4 | +100% |
| **代码行数** | ~500 | ~850 | +70% |
| **参数数量** | 11 | 17 | +55% |

### 详细分析

查看完整的实验分析报告：
- **失败分析**: `test/categorical_experiment/report/V2_FAILURE_ANALYSIS.md`
- **完整报告**: `test/categorical_experiment/FINAL_REPORT_V2.md`
- **可视化对比**: `test/categorical_experiment/report/comparison_v1_vs_v2.png`

## 关键发现与教训

### ✅ 成功经验 (来自 V1)

1. **简单即美**: 双组件设计（信息增益 + 覆盖度）足够有效
2. **稳定权重**: 固定的多样性权重（0.5）优于动态调整
3. **渐进策略**: 交互效应权重动态增加（0.5→3.0）符合学习过程
4. **最小距离覆盖**: 有效防止局部过采样

### ❌ 失败教训 (来自 V2)

1. **过度设计**: 添加过多组件增加调参复杂度而无实质收益
2. **软惩罚失效**: 0.01 的重复惩罚无法有效阻止重复采样
3. **分箱陷阱**: 分箱管理导致在中分区域的过度采样
4. **动态权重风险**: 多样性权重从 0.8 降到 0.2 导致后期过度开发

### 💡 核心见解

- **问题诊断**: V1 的唯一问题是重复采样，不是设计框架本身
- **解决方案**: 应使用硬排除（`-inf`），而非添加复杂机制
- **设计哲学**: 针对性修复优于全面重构
- **实验价值**: 负面结果也是宝贵的知识 - "The only real mistake is the one from which we learn nothing."

## 数学公式

### V1 采集函数

**α(x; D_t) = α_info(x; D_t) + α_cov(x; D_t)**

其中：

- **α_info**: 信息增益项
  - α_info = (1/|J|) Σ_j ΔVar[θ_j] + λ_t(r_t) × (1/|I|) Σ_{j,k} ΔVar[θ_jk]
  - 主效应 + 动态加权的交互效应
  
- **α_cov**: 空间覆盖项  
  - α_cov = γ × COV(x; D_t)
  - 基于 Gower 距离到现有样本
  - 使用 min_distance 方法

- **λ_t(r_t)**: 动态权重函数
  - 随着主效应不确定性降低，增加交互效应权重
  - 分段线性函数，依赖相对方差 r_t

### V2 采集函数（不推荐）

**α(x; D_t) = w_info × α_info + w_div × α_div + w_bin × α_bin + w_expl × α_expl**

其中添加了分箱管理和探索奖励，但增加了复杂度且效果不佳。

## 参数说明

### V1 关键参数

| 参数 | 描述 | 默认值 | 范围 |
|------|------|--------|------|
| `lambda_main` | 主效应权重 | 1.0 | [0, ∞) |
| `lambda_inter` | 交互效应初始权重 | 0.5 | [0, ∞) |
| `lambda_inter_max` | 交互效应最大权重 | 3.0 | [lambda_inter, ∞) |
| `tau_main` | 主效应方差阈值 | 0.3 | [0, 1.0] |
| `tau_inter` | 交互效应方差阈值 | 0.5 | [0, 1.0] |
| `gamma_diversity` | 覆盖度权重 | 0.5 | [0, ∞) |
| `coverage_method` | 覆盖度计算方法 | 'min_distance' | min/mean/median |
| `interaction_terms` | 交互项列表 | [] | [(i,j), ...] |

## 未来方向

### 推荐：V3 最小化改进

保持 V1 的全部设计，仅添加硬重复排除：

```python
class ImprovedAcqfV3(VarianceReductionWithCoverageAcqf):
    """V3: V1 + 硬重复排除"""
    
    def _evaluate_numpy(self, X_candidates):
        # 使用 V1 的全部逻辑
        scores = super()._evaluate_numpy(X_candidates)
        
        # 唯一改变：硬排除已采样设计
        for i, x in enumerate(X_candidates):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                scores[i] = -np.inf  # 硬排除，而非软惩罚
        
        return scores
```

**预期改进**:
- 唯一设计数: 39 → 55-65 (+40-65%)
- 高分设计: 8 → 12-16 (+50-100%)
- 平均分数: 8.72 → 9.0-9.3 (+3-7%)

### 可选优化

1. **候选集预过滤**:
   - 80% 从未采样设计 + 20% 已采样设计
   - 减少评估开销
   
2. **自适应覆盖权重**:
   - 早期较高（探索）→ 后期较低（开发）
   - 但需谨慎，V2 的教训显示动态权重可能适得其反

## 测试

### 运行单元测试

```bash
# V1 快速验证
cd test/categorical_experiment
pixi run python -c "from acquisition_function_v1 import VarianceReductionWithCoverageAcqf; print('✓ V1 导入成功')"

# 完整实验测试
pixi run python run_categorical_experiment_v1.py
```

### 性能基准

基于 80 次试验、360 个设计空间的分类变量实验：
- **V1**: 10.8% 覆盖率，8 个高分设计，平均 8.72 分
- **V2**: 7.8% 覆盖率，4 个高分设计，平均 7.94 分

## 依赖项

- Python 3.8+
- NumPy
- SciPy
- ConfigParser (标准库)
- Matplotlib (用于可视化)
- AEPsych (主框架)

## 版本历史

- **V1.0** (2025-10-30): 基线版本，简单有效 ✅
- **V2.0** (2025-10-30): 实验版本，失败但有价值 ⚠️
- **V3.0** (计划中): 最小化改进，硬排除重复 🎯

## 贡献与反馈

本项目是 AEPsych 扩展的一部分。实验设计、实现与分析完成于 2025年10月30日。

### 相关文档

- **API 文档**: `docs/README.md`
- **快速入门**: `docs/QUICKSTART.md`
- **实现细节**: `docs/IMPLEMENTATION_SUMMARY.md`
- **V2 失败分析**: `test/categorical_experiment/report/V2_FAILURE_ANALYSIS.md`
- **完整实验报告**: `test/categorical_experiment/FINAL_REPORT_V2.md`

## 许可证

遵循主项目 AEPsych 的许可证。

---

**重要提示**: 
- ✅ **推荐使用 V1 版本** - 简单、稳定、效果好
- ⚠️ **V2 仅供研究参考** - 失败实验，但提供了宝贵的教训
- 🎯 **V3 开发方向明确** - 最小化改动，硬排除重复

*"The only real mistake is the one from which we learn nothing." - Henry Ford*

# EUR (Expected Utility Reduction) 动态ANOVA采集函数 - 配置文档

## 📋 概览

本目录包含EUR采集函数的详细配置文档。EUR是一个模块化的期望效用削减采集函数，支持：

- ✅ 任意阶数交互（主效应 + 二阶 + 三阶 + ...）
- ✅ 动态权重自适应（λ_t、γ_t）
- ✅ 混合变量类型（分类、整数、连续）
- ✅ 模型稳定性追踪（SPS）
- ✅ 防守性覆盖机制

## 📁 配置文件导览

### 1. **QUICKSTART.ini** ⭐ 从这里开始

**对象**: 初学者和快速参考用户
**内容**:

- 三种常见场景的推荐配置（初级/进阶/高级）
- 常见问题与调优建议
- 性能基准数据
- 从其他采集函数的迁移指南

**何时使用**:

```
第一次使用 → QUICKSTART
遇到具体问题 → QUICKSTART的"常见问题"部分
想了解细节 → 对应的专项配置文件
```

**示例**:

```ini
# 心理物理实验（序数响应）→ 参考InitialConfig_Ordinal
[InitialConfig_Ordinal]
enable_main = true
enable_threeway = true
lambda_3 = 0.6
# ...
```

---

### 2. **EURAnovaMultiAcqf.ini** 核心采集函数

**对象**: EUR采集函数的完整参数配置
**主要部分**:

| 部分 | 参数数 | 说明 |
|------|--------|------|
| 核心功能开关 | 3 | enable_main, enable_pairwise, enable_threeway |
| 交互对配置 | 2 | interaction_pairs, interaction_triplets |
| 权重参数 | 5 | main_weight, lambda_2, lambda_3等 |
| 动态λ系统 | 5 | use_dynamic_lambda, tau1, tau2, lambda_min等 |
| 分段λ系统 | 4 | 可选：按实验阶段调整权重 |
| 覆盖度γ系统 | 8 | gamma, use_dynamic_gamma, gamma_max等 |
| SPS参数 | 3 | use_sps, sps_sensitivity, sps_ema_alpha |
| 安全机制 | 2 | tau_safe, gamma_penalty_beta |
| 融合方式 | 1 | additive或multiplicative |
| 局部采样 | 3 | local_jitter_frac, local_num, random_seed |
| 混合扰动 | 3 | use_hybrid_perturbation等 |
| 自动local_num | 2 | auto_compute_local_num, auto_local_num_max |
| 变量类型 | 1 | variable_types_list |
| Gower权重 | 1 | ard_weights |
| 调试 | 1 | debug_components |

**核心参数速查**:

```
启用交互？         → enable_pairwise, enable_threeway
权重如何变化？     → use_dynamic_lambda, tau1, tau2
覆盖度如何变化？   → use_dynamic_gamma, tau_n_min, tau_n_max
扰动如何生成？     → local_num, local_jitter_frac
```

---

### 3. **DynamicWeightEngine.ini** 权重引擎详解

**对象**: λ_t和γ_t动态权重系统的细节
**主要部分**:

| 系统 | 说明 | 关键参数 |
|------|------|---------|
| λ_t (二阶权重) | 基于参数方差收敛率r_t的动态权重 | use_dynamic_lambda, tau1, tau2, lambda_min/max |
| γ_t (覆盖度权重) | 基于样本数n_t的动态权重 | use_dynamic_gamma, tau_n_min/max, gamma_min/max |
| SPS系统 | 骨架点预测稳定性追踪 | use_sps, sps_sensitivity, sps_ema_alpha |
| 防守机制 | 低r_t时增强覆盖 | tau_safe, gamma_penalty_beta |
| 预算自适应 | 自动配置参数 | total_budget |

**数学公式** (在文件中有完整说明):

```
λ_target(r_t) = λ_max / (1 + exp(-τ1·(r_t - 0.5)))
λ_t^{new} = (1-τ2)·λ_t^{old} + τ2·λ_target
γ_t = γ_min + (γ_max - γ_min)·f(n_t, τ_n_min, τ_n_max)
```

**何时修改**:

```
权重变化太快/太慢    → 调整tau2
早期交互探索不足    → 增大lambda_2_init或减小tau_n_min
后期收敛太快        → 增大lambda_min
覆盖度权重太低      → 增大gamma或gamma_max
```

---

### 4. **LocalSampler.ini** 局部采样配置

**对象**: 围绕候选点生成扰动变体的方法
**主要部分**:

| 功能 | 说明 | 关键参数 |
|------|------|---------|
| 扰动幅度 | 高斯扰动的标准差 | local_jitter_frac |
| 样本数 | 每个候选点的变体数 | local_num (手动) 或 auto_compute_local_num (自动) |
| 混合扰动 | 对低水平离散变量穷举 | use_hybrid_perturbation, exhaustive_level_threshold |
| 变量类型 | 各维度的特征类型 | variable_types_list |
| Gower权重 | 混合距离中的维度权重 | ard_weights, coverage_method |

**分类vs连续vs整数**:

```
categorical  → 从历史值中离散采样
continuous   → 高斯扰动
integer      → 高斯扰动后舍入

混合扰动时：
  ≤threshold水平 → 穷举所有组合（精确）
  >threshold水平 → 高斯采样（快速）
```

**何时修改**:

```
覆盖不足         → 增大local_num或local_jitter_frac
计算成本太高     → 减小local_num或启用use_hybrid
有分类变量       → 设置variable_types_list和coverage_method
```

---

## 🚀 快速开始

### 最小化配置（仅需EURAnovaMultiAcqf.ini）

```ini
[EURAnovaMultiAcqf]
enable_main = true
enable_pairwise = true
lambda_2 = None              # 动态权重
use_dynamic_lambda = true
gamma = 0.3
local_num = 4
```

### 推荐完整配置（使用所有文件）

1. **复制模板**:

   ```bash
   cp QUICKSTART.ini my_experiment.ini
   # 编辑my_experiment.ini中的InitialConfig_*部分
   ```

2. **加载配置**:

   ```python
   from aepsych.config import Config
   from dynamic_eur_acquisition.eur_anova_multi import EURAnovaMultiAcqf
   
   config = Config.from_file("my_experiment.ini")
   acqf_opts = EURAnovaMultiAcqf.get_config_options(config, "EURAnovaMultiAcqf")
   acqf = EURAnovaMultiAcqf(model=model, **acqf_opts)
   ```

---

## 📊 参数对应关系

```
EURAnovaMultiAcqf
├── enable_main, enable_pairwise, enable_threeway
│   └── interaction_pairs, interaction_triplets
├── DynamicWeightEngine
│   ├── use_dynamic_lambda (λ_t系统)
│   │   ├── tau1, tau2
│   │   └── lambda_min, lambda_max, lambda_2_init
│   ├── use_dynamic_gamma (γ_t系统)
│   │   ├── tau_n_min, tau_n_max
│   │   └── gamma_min, gamma_max
│   ├── use_sps (SPS追踪)
│   │   ├── sps_sensitivity
│   │   └── sps_ema_alpha
│   └── total_budget (自动配置)
└── LocalSampler
    ├── local_num, local_jitter_frac, random_seed
    ├── use_hybrid_perturbation
    │   ├── exhaustive_level_threshold
    │   └── exhaustive_use_cyclic_fill
    ├── variable_types_list
    └── ard_weights
```

---

## 🔍 场景选择指南

### 我的问题是

| 场景 | 推荐起点 | 关键调整 |
|------|---------|---------|
| 连续回归，简单空间 | QUICKSTART::InitialConfig_Regression | lambda_max=0.8, gamma=0.3 |
| 序数响应(1-5评分) | QUICKSTART::InitialConfig_Ordinal | lambda_3=0.6, enable_threeway=true |
| 混合变量(分类+连续) | QUICKSTART::InitialConfig_Hybrid | use_hybrid_perturbation=true |
| 大设计空间(>10维) | EURAnovaMultiAcqf::auto_compute_local_num | =true |
| 小预算(<30试) | InitialConfig_Regression | gamma=0.4, total_budget=20 |
| 需要完全可复现 | 所有::random_seed | =42 (固定值) |
| 需要快速反馈 | 所有::debug_components | =true |

---

## ⚙️ 常见调优操作

### 问题：采样太集中（探索不足）

```ini
# 增加覆盖度权重
gamma = 0.4-0.5              # 从0.3提升
gamma_max = 0.6-0.7          # 从0.5提升
use_dynamic_gamma = true

# 弱化交互探索
lambda_min = 0.05            # 从0.1降低
enable_threeway = false      # 禁用三阶
```

### 问题：采样太分散（利用不足）

```ini
# 减少覆盖度权重
gamma = 0.1-0.2              # 从0.3降低
gamma_max = 0.3-0.4          # 从0.5降低

# 强化交互探索
lambda_2_init = 0.4-0.5      # 初期更激进
enable_threeway = true       # 启用三阶
tau1 = 0.8-1.0               # 快速响应
```

### 问题：计算太慢

```ini
# 减少局部采样
local_num = 2-3              # 从4降到2-3

# 减少交互
enable_threeway = false      # 禁用三阶
interaction_pairs = "0,1; 1,2"  # 手动指定少数对

# 禁用混合扰动
use_hybrid_perturbation = false

# 快速关闭
debug_components = false
```

### 问题：模型不稳定（预测波动）

```ini
# 增强EMA平滑
tau2 = 0.4-0.5               # 从0.3提升

# 监控收敛
use_sps = true               # 启用SPS
sps_ema_alpha = 0.8          # 从0.7提升（更平滑）

# 弱化交互
lambda_max = 0.6-0.7         # 从1.0降低
lambda_3 = 0.3-0.4           # 从0.5降低
```

---

## 📖 详细参考

每个配置文件都包含：

- **参数说明**: 类型、默认值、范围
- **公式解释**: 数学定义和直观理解
- **场景指导**: 何时使用、何时修改
- **交互影响**: 与其他参数的关系
- **最佳实践**: 推荐值和反面教训

**示例**:

```ini
# Gamma最大值
# 类型: float
# 默认: 0.5
# 范围: 0.3 - 1.0
# 说明: γ_t的上界...
# 直观理解: ...
# 推荐场景: ...
```

---

## 🔗 相关文件

- `../eur_anova_multi.py` - EUR采集函数主实现
- `../modules/` - 模块化组件（ANOVA、权重、采样等）
- `../docs/` - 理论文档和论文
- `../examples/` - 使用示例脚本

---

## ❓ 常见问题

**Q: 我应该先改什么参数?**
A: 从QUICKSTART开始，选择最接近你的场景的预设配置，然后仅修改与"问题"对应的几个参数。

**Q: 所有参数都要手动配置吗?**
A: 不需要。大多数参数有合理的默认值。只需配置：

- enable_main, enable_pairwise, enable_threeway （功能）
- lambda_2, gamma （权重）
- local_num （采样）
- variable_types_list （如有分类变量）

**Q: 配置如何保存到文件?**
A: 创建.ini文件，复制任一配置模板，修改参数。通过`Config.from_file()`加载。

**Q: 如何验证我的配置?**
A: 检查清单在QUICKSTART的"验证配置的检查清单"部分。

---

## 📝 版本历史

- **v1.0** (2025-12-07): 初始发布
  - 完整的EURAnovaMultiAcqf配置说明
  - DynamicWeightEngine权重系统详解
  - LocalSampler扰动采样详解
  - QUICKSTART快速开始指南

---

## 🎓 学习路径

1. **新手** (15分钟)
   - 阅读本README
   - 快速阅读QUICKSTART的"初级"部分
   - 复制InitialConfig_Regression

2. **进阶** (30分钟)
   - 阅读QUICKSTART的"场景对应速查表"
   - 浏览EURAnovaMultiAcqf.ini的相关部分
   - 调整参数进行实验

3. **高级** (1-2小时)
   - 详细阅读DynamicWeightEngine.ini
   - 理解λ_t和γ_t的数学原理
   - 研究SPS系统和防守机制
   - 设计自定义配置

---

欢迎使用EUR采集函数！有问题请参考对应的配置文件或提交Issue。

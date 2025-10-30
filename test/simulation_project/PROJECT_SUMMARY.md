# 模拟实验项目完成总结

## 项目概述

成功创建了一个完整的主动学习模拟实验项目,展示了 `VarianceReductionWithCoverageAcqf` 采集函数在虚拟被试实验中的应用。

## 项目结构

```
test/simulation_project/
├── virtual_subject.py          # 虚拟被试模拟器 (~270行)
├── run_experiment.py           # 完整实验流程 (~610行)
├── experiment_config.ini       # AEPsych标准配置
├── analyze_results.py          # 结果分析脚本 (~190行)
├── README.md                   # 完整文档
└── results/                    # 输出目录
    ├── experiment_data_*.csv           # 试次数据
    ├── training_data_*.npz             # 训练数据
    ├── experiment_metadata_*.json      # 元数据
    ├── evaluation_results_*.npz        # 评估结果
    ├── experiment_results_*.png        # 12面板可视化
    └── learning_curves_*.png           # 学习曲线分析
```

## 核心组件

### 1. 虚拟被试 (`virtual_subject.py`)

**功能**:

- 模拟真实心理物理学被试的响应行为
- 支持4种函数类型:
  - `linear`: 纯线性主效应
  - `nonlinear`: 非线性主效应(对数、平方根)
  - `interaction`: 线性+交互效应
  - `nonlinear_interaction`: 非线性+交互效应(最真实)

**特性**:

- 观测噪声: σ=0.3 (可配置)
- 响应类型: 连续值或二值判断
- 试次历史记录
- 模拟反应时
- 支持多被试池(`SubjectPool`)

**心理物理学真实性**:

```python
# 实际使用的函数(nonlinear_interaction)
y = 3.5√x₁ + 2.5log(1+5x₂) + 1.8x₃^0.7  # 非线性主效应
  + 2.0x₁^0.5 × x₂^0.6                  # 强度×持续时间交互
  - 0.8x₃(x₁+x₂)/2                      # 噪声抑制效应
  + N(0, 0.3)                            # 观测噪声
```

### 2. 实验流程 (`run_experiment.py`)

**`ExperimentRunner` 类**:

核心方法:

1. `__init__()`: 初始化被试、配置、采集函数
2. `_sobol_sample(n)`: 生成Sobol准随机样本
3. `run_initialization_phase(n_init=20)`: 阶段1 - Sobol初始化
4. `run_optimization_phase(n_opt=40)`: 阶段2 - 主动学习优化
5. `evaluate_model(n_test=1000)`: 测试集评估
6. `save_results()`: 保存多种格式数据
7. `visualize_results()`: 生成12面板图表

**实验设计**:

- 阶段1 (初始化): 20次Sobol采样 → 均匀覆盖参数空间
- 阶段2 (优化): 40次主动学习 → 自适应选择最有价值的点
- 评估: 1000个测试点 → 全面评估模型精度

**数据保存**:

- CSV: 试次级别数据,包含所有变量和时间戳
- NPZ: NumPy格式,训练数据和评估结果
- JSON: 元数据和配置信息
- PNG: 可视化结果(两个图表)

### 3. 配置文件 (`experiment_config.ini`)

**符合AEPsych标准格式**:

```ini
[common]
parnames = [intensity, duration, noise_level]
outcome_types = [continuous]

[intensity], [duration], [noise_level]
par_type = continuous
lower_bound = 0
upper_bound = 1

[init_strat]
min_asks = 20
generator = SobolGenerator

[opt_strat]
min_asks = 40
generator = OptimizeAcqfGenerator
model = GPRegressionModel
acqf = VarianceReductionWithCoverageAcqf

[VarianceReductionWithCoverageAcqf]
lambda_min = 0.3
lambda_max = 2.5
tau_1 = 0.6
tau_2 = 0.2
gamma = 0.6
interaction_terms = (0,1);(1,2)
```

### 4. 结果分析 (`analyze_results.py`)

**自动分析功能**:

1. **自适应行为分析**: 比较初始化和优化阶段
2. **探索-开发权衡**: 分析不同λ_t水平下的行为
3. **参数空间覆盖**: 统计采样分布
4. **性能对比**: 与随机采样的理论对比
5. **学习曲线可视化**: 4面板图表

**输出示例**:

```
自适应行为分析
初始化阶段 (n=20):
  - 响应均值: 6.894 ± 1.819
  - 响应范围: [3.658, 9.794]

优化阶段 (n=40):
  - 响应均值: 6.345 ± 2.544
  - λ_t 变化: 0.300 → 2.000  (从探索到开发)
  - r_t 变化: 0.826 → 0.291  (不确定性降低)

探索-开发权衡
探索主导 (n=12): λ_t=0.445, r_t=0.632
平衡阶段 (n=5):  λ_t=1.307, r_t=0.417
开发主导 (n=23): λ_t=1.821, r_t=0.323
```

## 实验结果

### 运行命令

```bash
cd test/simulation_project
pixi run python run_experiment.py
```

### 实际性能 (2025-10-29运行)

**实验配置**:

- 总试次: 60 (20初始化 + 40优化)
- 被试类型: nonlinear_interaction
- 观测噪声: σ=0.3
- 评估: 1000测试点

**最终性能**:

- **R² = 0.9332** (非常好的拟合)
- **MSE = 0.2261**
- **MAE = 0.4024**
- **相关系数 = 0.9758**

**自适应行为**:

- λ_t演化: 0.300 → 2.000
  - 早期(探索主导): λ≈0.3-0.8,高不确定性区域
  - 后期(开发主导): λ≈1.5-2.0,精细化交互建模
- r_t演化: 0.826 → 0.291
  - 相对方差持续下降,模型越来越精确
- 采集得分: 0.1734 → 0.0732
  - 得分降低表明参数空间逐渐被充分探索

**观测误差**:

- 初始化阶段: 0.184 ± 0.160
- 优化阶段: 0.252 ± 0.146
- 说明: 优化阶段误差略高是因为探索了更极端的区域

## 项目特点

### 1. 生产级质量

- 完整的错误处理
- 详细的文档字符串
- 类型提示
- 进度报告
- 结果验证

### 2. 科学严谨性

- 固定随机种子(可重复)
- 保存完整配置
- 记录真实值(用于验证)
- 标准评估指标
- 可视化诊断

### 3. 易于使用

- 一行命令运行
- 清晰的输出
- 自动保存结果
- 详细的README
- 分析脚本

### 4. 高度可扩展

- 模块化设计
- 易于修改参数
- 支持不同函数类型
- 可添加新的分析
- 可用于对比研究

## 使用场景

### 1. 演示展示

- 向他人展示采集函数的工作原理
- 理解探索-开发权衡
- 可视化自适应行为

### 2. 方法验证

- 测试新的采集函数变体
- 对比不同参数设置
- 验证理论预测

### 3. 教学用途

- 主动学习入门
- 贝叶斯优化原理
- 实验设计方法

### 4. 研究基础

- 作为baseline对比
- 扩展到更复杂场景
- 开发新的采集策略

## 扩展建议

### 短期扩展

1. **对比实验**: 与其他采集函数(UCB, EI)对比
2. **参数敏感性**: 系统性地测试不同参数组合
3. **多被试**: 使用`SubjectPool`模拟被试间变异

### 中期扩展

1. **自适应参数**: 根据r_t动态调整tau_1/tau_2
2. **不同噪声**: 测试对噪声水平的鲁棒性
3. **成本函数**: 添加不同区域的采样成本

### 长期扩展

1. **真实数据**: 适配到真实心理物理学数据
2. **多维拓展**: 测试4维以上参数空间
3. **在线学习**: 实时更新和可视化

## 总结

这是一个**完整、可运行、生产级**的模拟实验项目,完全符合用户需求:

✅ 虚拟被试根据真实心理物理函数响应  
✅ 使用标准AEPsych配置格式  
✅ 应用`VarianceReductionWithCoverageAcqf`采集函数  
✅ 实现完整的两阶段实验流程  
✅ 保存完整的实验数据  
✅ 提供详细的结果分析  
✅ 包含清晰的文档和使用说明  

**这个项目可以直接用于研究、演示和教学!**

---

最后修改: 2025-10-29  
状态: ✅ 完成并验证

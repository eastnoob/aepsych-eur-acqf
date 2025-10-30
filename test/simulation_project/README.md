# 完整模拟实验项目

这个目录包含一个完整的模拟心理物理学实验,使用标准AEPsych流程和我们的自定义采集函数。

## 项目结构

```
simulation_project/
├── virtual_subject.py       # 虚拟被试模拟器
├── experiment_config.ini     # 实验配置文件
├── run_experiment.py         # 主实验脚本
├── README.md                 # 本文件
└── results/                  # 实验结果(运行后生成)
    ├── experiment_data_*.csv        # 试次数据
    ├── training_data_*.npz          # 训练数据
    ├── experiment_metadata_*.json   # 元数据
    ├── evaluation_results_*.npz     # 评估结果
    └── experiment_results_*.png     # 可视化
```

## 实验设计

### 虚拟被试

**`virtual_subject.py`** 实现了一个虚拟被试,模拟真实的心理物理响应:

- **刺激参数** (3维):
  - `intensity` (x1): 刺激强度 [0, 1]
  - `duration` (x2): 刺激持续时间 [0, 1]
  - `noise_level` (x3): 背景噪声水平 [0, 1]

- **真实函数**:

  ```
  非线性主效应:
  y = 3.5*√x1 + 2.5*log(1+5*x2) + 1.8*x3^0.7
  
  交互效应:
  y += 2.0*x1^0.5*x2^0.6  (强度×持续时间)
  y -= 0.8*x3*(x1+x2)/2   (噪声抑制效应)
  ```

- **观测噪声**: σ = 0.3 (高斯噪声)

### 实验流程

遵循标准AEPsych两阶段流程:

**阶段1: 初始化 (n=20)**

- 使用Sobol准随机采样
- 覆盖整个参数空间
- 收集基础数据

**阶段2: 优化 (n=40)**

- 使用`VarianceReductionWithCoverageAcqf`选点
- 平衡信息增益和空间覆盖
- 动态调整主效应/交互效应权重

**总计**: 60次试验

### 采集函数配置

使用我们的自定义采集函数,配置为:

```ini
[VarianceReductionWithCoverageAcqf]
lambda_min = 0.3       # 早期:主效应优先
lambda_max = 2.5       # 后期:交互效应重要
tau_1 = 0.6            # 相对方差上阈值
tau_2 = 0.2            # 相对方差下阈值
gamma = 0.6            # 覆盖权重
interaction_terms = (0,1);(1,2)  # intensity×duration, duration×noise
```

## 运行实验

### 快速开始

```bash
cd test/simulation_project
pixi run python run_experiment.py
```

### 测试虚拟被试

```bash
pixi run python virtual_subject.py
```

## 输出文件

### 1. 试次数据 (`experiment_data_*.csv`)

每一行是一次试验:

| 列名 | 说明 |
|------|------|
| trial | 试次编号 |
| phase | 阶段(initialization/optimization) |
| x1, x2, x3 | 刺激参数 |
| response | 被试响应(观测值) |
| true_value | 真实值(仅用于分析) |
| lambda_t | 当前交互权重 |
| r_t | 相对方差 |
| acq_score | 采集函数分数 |
| timestamp | 时间戳 |

### 2. 训练数据 (`training_data_*.npz`)

NumPy格式,包含:

- `X_train`: 所有刺激参数 (n_trials, 3)
- `y_train`: 所有响应 (n_trials,)
- `param_names`: 参数名称

### 3. 元数据 (`experiment_metadata_*.json`)

JSON格式,包含:

- 实验配置
- 被试参数
- 采集函数参数
- 模型性能指标

### 4. 评估结果 (`evaluation_results_*.npz`)

测试集评估:

- `X_test`: 测试点
- `y_test_true`: 真实值
- `y_test_pred`: 预测值
- `metrics`: 性能指标

### 5. 可视化 (`experiment_results_*.png`)

12个子图的综合可视化:

1. **动态权重**: λ_t和r_t的演化
2. **采集分数**: 采集函数分数随时间变化
3. **响应演化**: 响应值随试次变化
4. **样本增长**: 累积样本数
5. **参数空间(X1-X2)**: intensity vs duration
6. **参数空间(X2-X3)**: duration vs noise
7. **响应分布**: 直方图
8. **真实vs观测**: 验证噪声模型
9. **预测准确度**: 模型预测vs真实值
10. **残差分析**: 预测误差
11. **残差分布**: 误差直方图
12. **性能统计**: MSE, R², 等指标

## 预期结果

基于我们之前的测试,预期:

- **R² > 0.95**: 优秀的拟合质量
- **MSE < 0.15**: 低预测误差
- **λ_t**: 从0.3增长到2.0+
- **r_t**: 从1.0降到0.2-0.4

## 实验特点

### 1. 真实性

- 模拟真实的心理物理函数(非线性+交互)
- 包含观测噪声
- 试次历史记录

### 2. 标准流程

- 遵循AEPsych配置格式
- 两阶段实验设计
- 标准评估方法

### 3. 完整数据

- 保存所有试次数据
- 包含真实值(用于分析)
- 记录采集函数状态

### 4. 可重复

- 固定随机种子
- 保存完整配置
- 可以重现结果

## 自定义实验

### 修改被试类型

编辑 `run_experiment.py`:

```python
exp = ExperimentRunner(
    subject_type="linear",  # 可选: linear, nonlinear, interaction, nonlinear_interaction
    subject_noise=0.2,      # 调整噪声水平
    ...
)
```

### 修改实验参数

编辑配置文件或直接修改代码:

```python
exp.run_initialization_phase(n_init=30)  # 更多初始样本
exp.run_optimization_phase(n_opt=50)     # 更多优化迭代
```

### 修改采集函数参数

编辑 `experiment_config.ini`:

```ini
[VarianceReductionWithCoverageAcqf]
gamma = 0.8  # 增加探索
lambda_max = 3.0  # 更强的交互效应权重
```

## 分析结果

### 自动分析

使用提供的分析脚本快速查看结果:

```bash
python analyze_results.py
```

这会输出:

- **自适应行为分析**: 初始化 vs 优化阶段统计
- **探索-开发权衡**: λ_t和r_t在不同阶段的行为
- **参数空间覆盖**: 采样分布的均匀性分析
- **学习曲线图**: 额外生成4面板分析图(`learning_curves_*.png`)

示例输出:

```
自适应行为分析
初始化阶段 (n=20):
  - 响应均值: 6.894 ± 1.819
  - λ_t 变化: N/A (未启用)

优化阶段 (n=40):
  - 响应均值: 6.345 ± 2.544
  - λ_t 变化: 0.300 → 2.000
  - r_t 变化: 0.826 → 0.291
  
探索-开发权衡
探索主导 (λ<0.8, n=12): r_t=0.632 ↔ 高不确定性区域
开发主导 (λ>1.5, n=23): r_t=0.323 ↔ 已探索充分
```

### 手动分析

如需自定义分析,可以加载数据:

```python
import pandas as pd
import numpy as np

# 加载试次数据
df = pd.read_csv('results/experiment_data_YYYYMMDD_HHMMSS.csv')

# 加载训练数据
data = np.load('results/training_data_YYYYMMDD_HHMMSS.npz')
X_train = data['X_train']
y_train = data['y_train']

# 加载评估结果
eval_data = np.load('results/evaluation_results_YYYYMMDD_HHMMSS.npz')
y_true = eval_data['y_test_true']
y_pred = eval_data['y_test_pred']
```

### 分析lambda演化

```python
import matplotlib.pyplot as plt

opt_data = df[df['phase'] == 'optimization']
plt.plot(opt_data['trial'], opt_data['lambda_t'])
plt.xlabel('Trial')
plt.ylabel('λ_t')
plt.title('Dynamic Weighting Evolution')
plt.show()
```

## 扩展

可能的扩展方向:

1. **多被试**: 使用`SubjectPool`模拟被试间变异
2. **自适应阈值**: 根据进度调整tau_1和tau_2
3. **不同函数**: 测试不同的心理物理模型
4. **对比实验**: 比较不同采集函数的效果

## 问题排查

**导入错误**:

- 确保从项目根目录运行
- 检查sys.path设置

**配置文件错误**:

- 验证INI文件格式
- 检查参数名称

**结果不理想**:

- 增加初始样本数
- 调整采集函数参数
- 检查被试噪声水平

---

**这是一个生产级的模拟实验,可以直接用于研究和演示!**

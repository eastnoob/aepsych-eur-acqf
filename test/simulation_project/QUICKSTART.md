# 快速开始指南

## 一条命令运行完整实验

```bash
cd extensions/dynamic_eur_acquisition/test/simulation_project
pixi run python run_experiment.py
```

## 查看结果分析

```bash
pixi run python analyze_results.py
```

## 输出文件位置

所有结果保存在 `results/` 目录:

- `experiment_data_*.csv` - 完整试次数据
- `experiment_results_*.png` - 12面板可视化
- `learning_curves_*.png` - 学习曲线分析
- `experiment_metadata_*.json` - 配置和性能指标
- 其他NPZ格式数据文件

## 预期结果

实验运行约2-3分钟,完成60次试验后输出:

- **R² ≈ 0.93** (高质量拟合)
- **λ_t**: 0.3 → 2.0 (探索到开发转换)
- **r_t**: 0.8 → 0.3 (不确定性降低)

## 自定义实验

### 修改被试类型

编辑 `run_experiment.py` 第580行:

```python
subject_type="linear"  # 可选: linear, nonlinear, interaction, nonlinear_interaction
```

### 修改试验次数

编辑 `run_experiment.py` 第582-583行:

```python
exp.run_initialization_phase(n_init=30)  # 默认20
exp.run_optimization_phase(n_opt=50)     # 默认40
```

### 修改采集函数参数

编辑 `experiment_config.ini` 的 `[VarianceReductionWithCoverageAcqf]` 部分:

```ini
lambda_min = 0.5    # 增大 → 减少早期探索
gamma = 0.8         # 增大 → 更重视覆盖
```

## 完整文档

- **README.md** - 详细文档
- **PROJECT_SUMMARY.md** - 项目总结和设计说明

## 常见问题

**Q: 为什么优化阶段误差更大?**  
A: 因为采集函数会主动探索高不确定性区域,这些区域通常在参数空间边缘,噪声影响更明显。

**Q: 如何提高R²?**  
A: 增加初始化样本(n_init)或优化次数(n_opt),或减少被试噪声(subject_noise)。

**Q: lambda_t为什么会变化?**  
A: 这是采集函数的核心特性 - 根据相对方差r_t动态调整探索/开发权重。

---

**开始探索吧!**

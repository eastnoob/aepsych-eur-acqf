# Dynamic EUR Acquisition Function

用于AEPsych框架的自适应采集函数，专门优化分类变量和混合类型变量的贝叶斯优化。

## 核心特性

- **ANOVA效应分解**: 通过主效应和交互效应最大化信息增益
- **动态权重调整**: 基于训练进度自动调节探索-利用平衡
- **混合变量支持**: 同时处理连续、离散和分类变量
- **覆盖优化**: 基于Gower距离的空间多样性保证

## 快速开始

### 安装依赖
```bash
pip install aepsych torch botorch gpytorch
```

### 基本使用
```python
from extensions.dynamic_eur_acquisition import EURAnovaMultiAcqf
from aepsych.server import AEPsychServer

# 配置参数
config = """
[EURAnovaMultiAcqf]
enable_main = true
enable_pairwise = true
use_dynamic_lambda = true
local_num = 6
variable_types_list = [categorical, continuous, integer]
"""

# 初始化服务器
server = AEPsychServer()
server.configure(config_str=config)

# 运行实验
for trial in range(25):
    next_x = server.ask()
    outcome = get_response(next_x)
    server.tell(config_str, outcome)
```

## 主要组件

### 采集函数版本
- **`EURAnovaMultiAcqf`**: 推荐版本，支持多阶交互和动态权重
- **`EURAnovaPairAcqf`**: 基础版本，仅支持二阶交互

### 核心模块
- `anova_effects.py`: ANOVA效应计算引擎
- `dynamic_weights.py`: 动态权重调整逻辑
- `local_sampler.py`: 混合变量扰动采样
- `coverage.py`: 空间覆盖度量

## 配置参数

### 基本配置
```ini
[EURAnovaMultiAcqf]
# 效应启用
enable_main = true          # 主效应
enable_pairwise = true      # 二阶交互
enable_threeway = false     # 三阶交互

# 权重参数
main_weight = 0.5          # 主效应权重
lambda_min = 0.1           # 最小交互权重
lambda_max = 1.0           # 最大交互权重

# 采样参数
local_num = 6              # 局部扰动点数
local_jitter_frac = 0.1    # 扰动幅度

# 变量类型
variable_types_list = [categorical, continuous, integer]
```

### 高级配置
```ini
[EURAnovaMultiAcqf]
# 动态权重
use_dynamic_lambda = true   # 启用动态λ调整
tau1 = 0.7                 # 收敛阈值上界
tau2 = 0.3                 # 收敛阈值下界

# 混合扰动策略
use_hybrid_perturbation = true
exhaustive_level_threshold = 3

# 覆盖计算
coverage_method = min_distance
gamma_max = 0.5
```

## 版本说明

- **v2.1.0**: 混合扰动策略，支持低基数分类变量优化
- **v2.0.0**: 多阶交互支持，动态权重引擎
- **v1.0.0**: 基础二阶交互实现

## 文档和示例

- 📖 [详细文档](docs/README_FULL_HISTORY.md)
- 🚀 [快速开始](docs/QUICK_REFERENCE.md)
- ⚙️ [配置指南](docs/⭐AEPSYCH_CONFIG_GUIDE.md)
- 📊 [实验报告](docs/FINAL_SUMMARY.md)

## 项目结构

```
extensions/dynamic_eur_acquisition/
├── eur_anova_multi.py      # 主采集函数
├── eur_anova_pair.py       # 二阶交互版本
├── modules/                # 核心模块
│   ├── anova_effects.py    # 效应计算
│   ├── dynamic_weights.py  # 权重调整
│   ├── local_sampler.py    # 扰动采样
│   └── coverage.py         # 覆盖计算
├── configs/                # 配置文件
├── examples/               # 使用示例
└── test/                   # 测试套件
```

## 贡献

欢迎提交Issue和Pull Request。请查看[贡献指南](CONTRIBUTORS.md)了解详细信息。

## 许可证

遵循AEPsych项目许可证。
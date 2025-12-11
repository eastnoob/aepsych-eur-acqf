# Dynamic EUR Acquisition Function

用于AEPsych框架的贝叶斯优化采集函数，专门为混合变量优化设计。

## 核心公式

$$
EUR(x) = Info(x) \oplus \gamma_t \times Coverage(x)
$$

其中：
- Info(x): 主效应 + λ_t × 交互效应
- Coverage(x): 基于Gower距离的覆盖度量
- λ_t, γ_t: 动态权重参数

## 使用方法

### 安装
```bash
pip install aepsych torch botorch gpytorch
```

### 基本使用
```python
from extensions.dynamic_eur_acquisition import EURAnovaMultiAcqf
from aepsych.server import AEPsychServer

config = """
[EURAnovaMultiAcqf]
enable_main = true
enable_pairwise = true
use_dynamic_lambda = true
local_num = 6
variable_types_list = [categorical, continuous, integer]
"""

server = AEPsychServer()
server.configure(config_str=config)

# 运行实验
for trial in range(25):
    next_x = server.ask()
    outcome = get_response(next_x)
    server.tell(config_str, outcome)
```

### 主要组件
- `EURAnovaMultiAcqf`: 推荐版本，支持多阶交互
- `anova_effects.py`: 效应计算
- `dynamic_weights.py`: 权重调整
- `local_sampler.py`: 混合变量采样

## 文档
- [详细文档](docs/README_FULL_HISTORY.md)
- [配置指南](docs/⭐AEPSYCH_CONFIG_GUIDE.md)
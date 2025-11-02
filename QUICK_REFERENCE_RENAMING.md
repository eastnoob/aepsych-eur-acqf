# 快速参考：EURAnovaPairAcqf 重命名完成

## 📦 文件映射

| 原文件 | 新文件 |
|--------|--------|
| `mc_anova_acquisition.py` | `eur_anova_pair_acquisition.py` |
| `MonteCarloAnovaAcqf` 类 | `EURAnovaPairAcqf` 类 |

## 🎯 导入更新

**旧导入方式（已过时）**

```python
from mc_anova_acquisition import MonteCarloAnovaAcqf
```

**新导入方式（推荐）**

```python
from eur_anova_pair_acquisition import EURAnovaPairAcqf
```

## ✅ 验证状态

- ✅ 文件已创建：`eur_anova_pair_acquisition.py`
- ✅ 类已重命名：`EURAnovaPairAcqf`
- ✅ 导入测试通过
- ✅ 功能测试通过
- ✅ 文档已更新

## 🧪 快速测试

```python
import torch
from eur_anova_pair_acquisition import EURAnovaPairAcqf
from botorch.models import SingleTaskGP

# 初始化数据与模型
X_train = torch.randn(10, 3)
y_train = torch.randn(10)
model = SingleTaskGP(X_train, y_train.unsqueeze(-1))

# 创建采集函数（使用新名称）
acqf = EURAnovaPairAcqf(
    model=model,
    interaction_pairs=[(0, 1)],
    variable_types={0: "continuous", 1: "categorical"}
)

# 计算采集值
X_test = torch.randn(5, 1, 3)
acq_values = acqf(X_test)
print(f"Acquisition values: {acq_values}")
```

## 📌 备注

- 原文件 `mc_anova_acquisition.py` 保留用于参考
- 所有新代码应使用 `eur_anova_pair_acquisition` 模块
- 名称反映设计：EUR (Expected Utility) + Anova (Decomposition) + Pair (Interactions)

---

**最后更新**: 2025-11-02
**状态**: ✅ 完成

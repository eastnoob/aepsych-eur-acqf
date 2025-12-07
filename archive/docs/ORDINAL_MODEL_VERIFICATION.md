# 序数模型配置验证指南

## 📌 为什么需要验证？

当使用Ordinal GP模型（如Likert量表响应）时，EUR采集函数的信息度量依赖于**cutpoints**（切分点）。如果cutpoints未正确学习或无法访问，采集函数将**降级到方差指标**，导致：

- ❌ 信息度量不准确（用方差代替熵）
- ❌ 交互探索效果下降
- ❌ 选点质量降低

## ✅ 验证时机

**必须在EUR采样开始前验证！**

```python
# ✅ 正确流程
# 1. Warmup阶段
for i in range(warmup_trials):
    next_x = server.ask()
    outcome = get_response(next_x)
    server.tell(config, outcome)

# 2. 验证序数模型配置（关键步骤！）
from tools.verify_ordinal_model import verify_ordinal_configuration

is_valid, diagnostics = verify_ordinal_configuration(
    server,
    min_training_samples=10  # warmup样本数应≥10
)

if not is_valid:
    raise RuntimeError("序数模型配置验证失败！检查上述警告。")

# 3. 开始EUR采样
for i in range(eur_trials):
    next_x = server.ask()  # 使用EUR采集函数
    ...
```

## 🔍 验证脚本详解

### 基础用法

```python
from tools.verify_ordinal_model import verify_ordinal_configuration

# 在warmup结束后调用
is_valid, diagnostics = verify_ordinal_configuration(
    server=server,
    min_training_samples=10,  # 最小训练样本数
    verbose=True               # 打印详细信息
)

print(f"验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
```

### 保存验证报告

```python
from tools.verify_ordinal_model import verify_and_save_report

is_valid = verify_and_save_report(
    server=server,
    output_path="ordinal_verification_report.json",
    min_training_samples=10
)
```

### 检查诊断信息

```python
is_valid, diagnostics = verify_ordinal_configuration(server)

# 检查关键字段
print(f"是否为Ordinal模型: {diagnostics['is_ordinal']}")
print(f"n_levels: {diagnostics['n_levels']}")
print(f"训练样本数: {diagnostics['n_train']}")
print(f"Cutpoints找到: {diagnostics['cutpoints_found']}")
print(f"Cutpoints值: {diagnostics['cutpoints_values']}")
print(f"熵计算测试: {'通过' if diagnostics['entropy_test_passed'] else '失败'}")

# 检查问题和警告
if diagnostics['issues']:
    print(f"\n❌ 发现问题:")
    for issue in diagnostics['issues']:
        print(f"  - {issue}")

if diagnostics['warnings']:
    print(f"\n⚠️  警告:")
    for warn in diagnostics['warnings']:
        print(f"  - {warn}")
```

## 🛠️ 常见问题排查

### 问题1：无法获取cutpoints

**症状**：
```
❌ 错误：无法获取cutpoints！
   检查过的属性名: ['cutpoints', 'cut_points', 'cut_points_', '_cutpoints']
```

**原因**：
1. OrdinalLikelihood未正确初始化
2. 使用了不兼容的AEPsych版本
3. 模型未训练

**解决**：
```python
# 检查配置文件
[common]
outcome_type = ordinal  # ← 确保设置为ordinal

[OrdinalGPModel]
n_levels = 5  # ← 确保设置n_levels
```

### 问题2：训练样本数不足

**症状**：
```
⚠️  训练样本数 (5) 少于建议值 (10)，cutpoints估计可能不稳定
```

**解决**：
```ini
[warmup_strategy]
n_trials = 15  # 增加warmup样本数（从10→15）
```

### 问题3：Cutpoints包含inf值

**症状**：
```
⚠️  Cutpoints包含inf值: tensor([-inf, 2.1, 2.8, 3.5])
   可能原因：某些水平从未被观测到
```

**原因**：
- 如果有5个水平但从未观测到y=1，优化器可能把c₁推到-∞

**解决**：
```python
# 方案1：增加warmup样本数，确保覆盖所有水平
# 方案2：使用随机采样确保早期覆盖
[warmup_strategy]
generator = SobolQMCNormalGenerator  # 而非EUR
n_trials = 15
```

### 问题4：熵值超过最大熵

**症状**：
```
⚠️  熵值 (1.85) 超过最大熵 (1.61)
   可能存在数值问题
```

**原因**：
- Cutpoints估计不稳定
- 概率计算数值误差

**解决**：
```python
# 这通常是警告而非错误，但建议：
# 1. 检查cutpoints是否单调递增
# 2. 增加训练样本数
# 3. 检查模型收敛性
```

## 📊 验证输出示例

### ✅ 成功案例

```
======================================================================
                      序数模型配置验证工具 v1.0
======================================================================

【步骤1】Likelihood类型检查
  ✓ Likelihood类型: OrdinalLikelihood

【步骤2】n_levels配置检查
  ✓ n_levels = 5
  ✓ 期望cutpoints数量 = 4

【步骤3】训练样本数检查
  ✓ 训练样本数: 15

【步骤4】Cutpoints可访问性检查
  ✓ 找到cutpoints (属性名: 'cutpoints')
  ✓ Cutpoints值: [-1.52  -0.48   0.51   1.49]

【步骤5】Cutpoints合理性检查
  ✓ Cutpoints数量正确: 4
  ✓ Cutpoints单调递增
  ✓ Cutpoints无inf值

【步骤6】熵计算功能性测试
  ✓ 熵计算测试通过
  ✓ 测试点熵值: H = 1.4523
  ✓ 熵值在合理范围 [0, 1.6094]

======================================================================
            ✅ 所有关键检查通过！序数模型配置正确。
======================================================================
```

### ⚠️ 警告案例

```
======================================================================
【步骤3】训练样本数检查
  ✓ 训练样本数: 8
  ⚠️  训练样本数 (8) 少于建议值 (10)，cutpoints估计可能不稳定

【步骤5】Cutpoints合理性检查
  ⚠️  Cutpoints包含inf值: [-inf 2.1 2.8 3.5]
     可能原因：某些水平从未被观测到
======================================================================
            ✅ 所有关键检查通过！序数模型配置正确。
⚠️  共有 2 个警告（不影响使用）:
   1. 训练样本数 (8) 少于建议值 (10)，cutpoints估计可能不稳定
   2. Cutpoints包含inf值: [-inf 2.1 2.8 3.5]
======================================================================
```

## 🎯 最佳实践

### 推荐的实验流程

```python
#!/usr/bin/env python3
"""推荐的EUR实验流程（包含序数模型验证）"""

from aepsych.server import AEPsychServer
from tools.verify_ordinal_model import verify_and_save_report

# 1. 初始化服务器
server = AEPsychServer()
server.configure(config_str=config_content)

# 2. Warmup阶段（建议≥10次）
print("=== Warmup阶段 ===")
for trial in range(15):  # 至少10次
    next_x = server.ask()
    outcome = get_subject_response(next_x)
    server.tell(config, outcome)
    print(f"Warmup {trial+1}/15: outcome={outcome}")

# 3. 验证序数模型配置（关键！）
print("\n=== 验证序数模型配置 ===")
is_valid = verify_and_save_report(
    server=server,
    output_path="results/ordinal_verification.json",
    min_training_samples=10
)

if not is_valid:
    raise RuntimeError(
        "❌ 序数模型配置验证失败！\n"
        "请检查上述警告，确保：\n"
        "1. OrdinalLikelihood正确初始化（n_levels设置）\n"
        "2. Warmup样本数≥10\n"
        "3. Cutpoints可访问且单调递增\n"
        "查看详细报告: results/ordinal_verification.json"
    )

# 4. EUR阶段
print("\n=== EUR采样阶段 ===")
for trial in range(25):
    next_x = server.ask()  # 使用EUR采集函数
    outcome = get_subject_response(next_x)
    server.tell(config, outcome)
    print(f"EUR {trial+1}/25: outcome={outcome}")

print("\n✓ 实验完成！")
```

### 自动化验证脚本

```python
def run_experiment_with_validation(config_path, n_warmup=15, n_eur=25):
    """
    运行实验，自动验证序数模型配置

    Args:
        config_path: 配置文件路径
        n_warmup: Warmup次数（建议≥10）
        n_eur: EUR采样次数
    """
    from aepsych.server import AEPsychServer
    from tools.verify_ordinal_model import verify_ordinal_configuration
    import warnings

    server = AEPsychServer()

    with open(config_path) as f:
        config_str = f.read()

    server.configure(config_str=config_str)

    # Warmup
    for i in range(n_warmup):
        next_x = server.ask()
        outcome = get_response(next_x)
        server.tell(config_str, outcome)

    # 自动验证
    is_valid, diag = verify_ordinal_configuration(
        server,
        min_training_samples=max(5, n_warmup // 2),
        verbose=False
    )

    if not is_valid:
        # 汇总所有问题
        issues_str = "\n".join(f"  - {issue}" for issue in diag['issues'])
        raise RuntimeError(f"序数模型配置验证失败:\n{issues_str}")

    if diag['warnings']:
        # 只是警告，打印但继续
        for warn in diag['warnings']:
            warnings.warn(warn)

    # EUR阶段
    for i in range(n_eur):
        next_x = server.ask()
        outcome = get_response(next_x)
        server.tell(config_str, outcome)

    return server

# 使用
server = run_experiment_with_validation(
    "config.ini",
    n_warmup=15,
    n_eur=25
)
```

## 📚 参考

- EUR采集函数实现: `eur_anova_pair.py`
- 序数熵计算: `modules/ordinal_metrics.py`
- 验证脚本源码: `../../tools/verify_ordinal_model.py`

---

**最后更新**: 2025-11-26
**作者**: EUR开发团队

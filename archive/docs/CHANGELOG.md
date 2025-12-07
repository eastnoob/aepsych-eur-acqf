# Changelog

All notable changes to the EUR ANOVA Acquisition Function will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (feature/hybrid-perturbation)

#### 混合扰动策略 (Hybrid Perturbation Strategy)

**版本**: v2.1.0 | **日期**: 2025-11-26

针对2-3水平离散变量场景的核心优化，显著提升效应发现质量。

**新增参数**:

- `use_hybrid_perturbation` (bool, 默认False): 启用混合扰动策略
- `exhaustive_level_threshold` (int, 默认3): 穷举水平数阈值
- `exhaustive_use_cyclic_fill` (bool, 默认True): 循环填充到local_num
- `auto_compute_local_num` (bool, 默认False): 自动计算local_num基于LCM（v2.1.1新增）
- `auto_local_num_max` (int, 默认12): 自动计算local_num的上限（v2.1.1新增）

**新增/修改模块**:

- `modules/local_sampler.py`: 独立的局部扰动采样器
  - `_perturb_categorical()`: 分类变量穷举逻辑
  - `_perturb_integer()`: 整数变量穷举逻辑
  - `_perturb_continuous()`: 连续变量扰动（保持不变）
  - `_compute_lcm()`: 计算最小公倍数（v2.1.1新增）
  - `_auto_compute_local_num()`: 自动计算local_num基于低水平变量（v2.1.1新增）

- `eur_anova_pair.py`: 主采集函数更新
  - `_make_local_hybrid()`: 集成混合扰动策略
  - 新增三个混合扰动参数传递

**新增配置文件**:

- `configs/hybrid_perturbation_optimized.ini`: 优化配置（启用混合扰动）
- `configs/baseline_config.ini`: 基线配置（用于对比实验）

**新增测试框架**:

- `tests/test_hybrid_perturbation_comparison.py`: 对比测试脚本框架
  - 支持多次重复实验（统计显著性检验）
  - 生成模拟数据（2-3水平变量）
  - 评估指标：R²、RMSE、效应检测准确率

**新增工具**:

- `tools/verify_ordinal_model.py`: 序数模型配置验证脚本
  - 验证cutpoints可访问性
  - 检查单调性和inf值
  - 熵计算功能性测试

**新增文档**:

- `docs/ORDINAL_MODEL_VERIFICATION.md`: 序数模型验证完整指南
  - 使用时机和步骤
  - 常见问题排查
  - 最佳实践
- `README.md`: 更新主文档，新增混合扰动策略章节

**核心改进逻辑**:

1. **穷举模式（≤3水平）**:
   - 分类变量：枚举所有unique值
   - 整数变量：枚举所有可能整数
   - 循环填充到`local_num`确保均衡覆盖
   - 例如：3水平 + local_num=6 → [0,1,2,0,1,2]

2. **高斯模式（>3水平）**:
   - 保持原有随机扰动逻辑
   - 避免穷举组合爆炸

3. **推荐配置**:
   - `local_num = 6` (2和3的LCM，确保均衡)
   - `use_hybrid_perturbation = True`
   - `exhaustive_level_threshold = 3`

**预期效果**:

- 效应发现能力提升10-20%（R²提高）
- 参数估计质量改善（RMSE降低）
- 完全覆盖所有离散水平（避免遗漏）

**向后兼容性**:

- 默认`use_hybrid_perturbation = False`，保持原有行为
- 不影响现有配置文件
- 仅新配置需要显式启用

**相关Issue/PR**:

- Branch: `feature/hybrid-perturbation`
- Base: `master`

---

## [2.0.0] - 2025-11-XX (假设的先前版本)

### Added

- Dynamic λ_t weight for interaction effects
- Dynamic γ_t weight for coverage/information balance
- Ordinal entropy calculation with numerical stability improvements
- Mixed variable type support (categorical, integer, continuous)
- Gower distance for coverage computation
- Parameter variance tracking for convergence monitoring

### Changed

- Refactored ANOVA effect decomposition for batch efficiency
- Improved categorical variable handling

### Fixed

- Numerical stability issues in ordinal entropy calculation
- Cutpoints accessibility for different AEPsych versions

---

## [1.0.0] - 2025-XX-XX (假设的初始版本)

### Added

- Initial EUR ANOVA acquisition function implementation
- Main effect and interaction effect decomposition
- Coverage-based exploration
- Basic variable type inference from transforms

---

## Development Notes

### Testing混合扰动策略

运行对比实验：

```bash
python tests/test_hybrid_perturbation_comparison.py
```

验证序数模型配置：

```python
from tools.verify_ordinal_model import verify_and_save_report

is_valid = verify_and_save_report(
    server=server,
    output_path="results/ordinal_verification.json"
)
```

### Breaking Changes

**无** - 当前更新完全向后兼容，默认行为不变。

### Migration Guide

#### 从原始配置迁移到混合扰动策略

**步骤1**: 更新配置文件

```ini
[EURAnovaPairAcqf]
# 原有参数保持不变...

# 新增混合扰动参数
use_hybrid_perturbation = True   # 启用混合扰动
exhaustive_level_threshold = 3   # 对≤3水平变量穷举
exhaustive_use_cyclic_fill = True  # 循环填充

# 推荐调整local_num
local_num = 6  # 从4增加到6（如果有2-3水平混合场景）
```

**步骤2**: （可选）验证序数模型

如果使用序数响应模型，在EUR采样前验证：

```python
from tools.verify_ordinal_model import verify_ordinal_configuration

is_valid, diagnostics = verify_ordinal_configuration(
    server,
    min_training_samples=10,
    verbose=True
)

if not is_valid:
    raise RuntimeError("序数模型配置验证失败！")
```

**步骤3**: 运行实验

无需修改实验代码，采集函数会自动使用新策略。

---

## Contributors

- eastnoob (2025-11-26): 混合扰动策略实现、序数模型验证工具
- AEPsych Team: 核心框架和原始EUR实现

---

## License

遵循主项目 AEPsych 的许可证。

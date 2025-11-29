"""
Dynamic EUR Acquisition Modules

模块化组件：
- anova_effects: ANOVA效应层次化计算引擎
- ordinal_metrics: 序数模型熵计算
- dynamic_weights: 动态权重自适应系统
- local_sampler: 混合变量局部扰动生成器
- coverage: 覆盖度计算（Gower距离）
- config_parser: 配置解析工具
- diagnostics: 诊断和调试工具
"""

from .anova_effects import (
    ANOVAEffect,
    MainEffect,
    PairwiseEffect,
    ThreeWayEffect,
    ANOVAEffectEngine,
    create_effects_from_config,
)

from .ordinal_metrics import OrdinalMetricsHelper

from .dynamic_weights import DynamicWeightEngine

from .local_sampler import LocalSampler

from .coverage import CoverageHelper

from .config_parser import (
    parse_interaction_pairs,
    parse_interaction_triplets,
    parse_variable_types,
    parse_ard_weights,
    validate_interaction_indices,
)

from .diagnostics import DiagnosticsManager


__all__ = [
    # ANOVA效应
    "ANOVAEffect",
    "MainEffect",
    "PairwiseEffect",
    "ThreeWayEffect",
    "ANOVAEffectEngine",
    "create_effects_from_config",
    # 序数模型
    "OrdinalMetricsHelper",
    # 动态权重
    "DynamicWeightEngine",
    # 局部采样
    "LocalSampler",
    # 覆盖度
    "CoverageHelper",
    # 配置解析
    "parse_interaction_pairs",
    "parse_interaction_triplets",
    "parse_variable_types",
    "parse_ard_weights",
    "validate_interaction_indices",
    # 诊断
    "DiagnosticsManager",
]

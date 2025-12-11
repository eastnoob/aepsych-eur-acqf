#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for Ordinal Parameter Extension

测试覆盖：
1. LocalSampler 与 ordinal 类型的集成
2. 变量类型推断
3. EUR-ANOVA 采集函数兼容性
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# 添加 extensions 到 path
project_root = Path(__file__).parent.parent.parent.parent
ext_path = project_root / "extensions" / "dynamic_eur_acquisition"
if str(ext_path) not in sys.path:
    sys.path.insert(0, str(ext_path))

from modules.local_sampler import LocalSampler
from transforms.ops.custom_ordinal import CustomOrdinal


class TestLocalSamplerOrdinal:
    """LocalSampler 与 ordinal 集成测试"""

    def test_ordinal_perturbation_basic(self):
        """测试 ordinal 扰动基础功能"""
        # 创建 ordinal transform
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [2.0, 2.5, 3.5]}
        )

        # 规范化值
        normalized_vals = ordinal.normalized_values[0]  # [0.0, 0.333..., 1.0]

        # 创建 sampler
        sampler = LocalSampler(
            variable_types={0: "ordinal"},
            local_jitter_frac=0.1,
            local_num=4,
            random_seed=42
        )

        # 准备训练数据（使用精确的规范化值）
        X_train_normalized = np.array([
            [normalized_vals[0], 0.5],
            [normalized_vals[1], 0.3],
            [normalized_vals[2], 0.8]
        ])

        sampler.update_data(X_train_normalized)

        # 候选点（规范化值）
        X_can_t = torch.tensor([[normalized_vals[1], 0.5]], dtype=torch.float32)

        # 扰动
        X_perturbed = sampler.sample(X_can_t, dims=[0])

        # 验证形状
        assert X_perturbed.shape == (1 * 4, 2)

        # 验证第一维（ordinal）的值都在规范化值集合中
        perturbed_vals_dim0 = X_perturbed[:, 0].cpu().numpy()
        for val in perturbed_vals_dim0:
            # 检查是否接近某个规范化值
            distances = np.abs(normalized_vals - val)
            assert np.min(distances) < 1e-5

        # 验证第二维（连续）未改变（未扰动）
        assert torch.allclose(X_perturbed[:, 1], torch.tensor(0.5, dtype=torch.float32))

    def test_mixed_variable_types(self):
        """测试混合变量类型：ordinal + continuous + categorical"""
        sampler = LocalSampler(
            variable_types={
                0: "custom_ordinal",
                1: "continuous",
                2: "categorical"
            },
            local_jitter_frac=0.1,
            local_num=4,
            random_seed=42
        )

        # 训练数据
        X_train = np.array([
            [0.0, 0.5, 1.0],
            [0.5, 0.3, 2.0],
            [1.0, 0.8, 1.0]
        ])

        sampler.update_data(X_train)

        # 候选点
        X_can = torch.tensor([[0.5, 0.5, 1.0]], dtype=torch.float32)

        # 扰动所有维度
        X_perturbed = sampler.sample(X_can, dims=[0, 1, 2])

        assert X_perturbed.shape == (1 * 4, 3)

        # 验证 ordinal 维（第0维）约束到训练数据中的值
        unique_ordinal = np.unique(X_train[:, 0])
        perturbed_ordinal = X_perturbed[:, 0].cpu().numpy()
        for val in perturbed_ordinal:
            assert np.min(np.abs(unique_ordinal - val)) < 1e-5

        # 验证 categorical 维（第2维）从训练数据采样
        unique_cat = np.unique(X_train[:, 2])
        perturbed_cat = X_perturbed[:, 2].cpu().numpy()
        for val in perturbed_cat:
            assert val in unique_cat

    def test_hybrid_perturbation_exhaustive(self):
        """测试混合扰动策略：低水平穷举"""
        sampler = LocalSampler(
            variable_types={0: "ordinal"},
            local_jitter_frac=0.1,
            local_num=6,
            use_hybrid_perturbation=True,
            exhaustive_level_threshold=3,
            exhaustive_use_cyclic_fill=True,
            random_seed=42
        )

        # 3水平 ordinal 变量
        X_train = np.array([
            [0.0, 0.5],
            [0.5, 0.3],
            [1.0, 0.8]
        ])

        sampler.update_data(X_train)

        X_can = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

        # 扰动
        X_perturbed = sampler.sample(X_can, dims=[0])

        # 验证穷举：应该循环包含所有3个水平
        perturbed_vals = X_perturbed[:, 0].cpu().numpy()
        unique_vals = np.unique(X_train[:, 0])

        # 检查是否覆盖所有水平
        for val in unique_vals:
            assert np.any(np.abs(perturbed_vals - val) < 1e-5)

    def test_ordinal_perturbation_seeded(self):
        """测试随机种子可复现性"""
        sampler1 = LocalSampler(
            variable_types={0: "ordinal"},
            local_jitter_frac=0.1,
            local_num=4,
            random_seed=123
        )

        sampler2 = LocalSampler(
            variable_types={0: "ordinal"},
            local_jitter_frac=0.1,
            local_num=4,
            random_seed=123
        )

        X_train = np.array([[0.0], [0.5], [1.0]])
        sampler1.update_data(X_train)
        sampler2.update_data(X_train)

        X_can = torch.tensor([[0.5]], dtype=torch.float32)

        # 两次扰动应该相同（相同种子）
        X_perturbed1 = sampler1.sample(X_can, dims=[0])
        X_perturbed2 = sampler2.sample(X_can, dims=[0])

        assert torch.allclose(X_perturbed1, X_perturbed2)


class TestVariableTypeInference:
    """变量类型推断测试（需要 mock model）"""

    def test_config_parser_ordinal_recognition(self):
        """测试 config_parser 识别 ordinal 类型"""
        from modules.config_parser import parse_variable_types

        # 测试不同格式
        vt1 = parse_variable_types(["ordinal", "continuous"])
        assert vt1[0] == "ordinal"
        assert vt1[1] == "continuous"

        vt2 = parse_variable_types(["custom_ordinal", "integer"])
        assert vt2[0] == "custom_ordinal"
        assert vt2[1] == "integer"

        vt3 = parse_variable_types(["custom_ordinal_mono", "categorical"])
        assert vt3[0] == "custom_ordinal_mono"
        assert vt3[1] == "categorical"

        # 测试缩写
        vt4 = parse_variable_types(["ord", "cat"])
        assert vt4[0] == "ordinal"
        assert vt4[1] == "categorical"

    def test_infer_variable_types_from_transform(self):
        """测试 EUR 是否能从 CustomOrdinal transform 推断变量类型"""
        from types import SimpleNamespace
        from eur_anova_pair import EURAnovaPairAcqf

        # 创建 CustomOrdinal transform
        ordinal = CustomOrdinal(indices=[0], values={0: [2.0, 2.5, 3.5]})

        # 模拟模型，包含 transforms 与训练数据
        dummy_model = SimpleNamespace()
        dummy_model.transforms = {"ord": ordinal}
        import torch
        dummy_model.train_inputs = [torch.tensor([[2.0], [2.5], [3.5]], dtype=torch.float32)]
        dummy_model.train_targets = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)

        acqf = EURAnovaPairAcqf(model=dummy_model)

        # 触发数据同步，以调用 _maybe_infer_variable_types
        acqf._ensure_fresh_data()

        assert isinstance(acqf.variable_types, dict)
        assert acqf.variable_types[0] == "ordinal"


class TestEURAnovaCompatibility:
    """EUR-ANOVA 采集函数兼容性测试（简化版）"""

    def test_ordinal_unique_vals_extraction(self):
        """测试 ordinal 的 unique 值提取"""
        # 模拟从训练数据提取 unique 值
        X_train = np.array([
            [0.0, 0.5],
            [0.333, 0.3],
            [1.0, 0.8],
            [0.0, 0.6],
            [0.333, 0.4]
        ])

        # 提取第一维的 unique 值
        unique_vals = np.unique(X_train[:, 0])

        # 应该是规范化后的 ordinal 值
        expected = np.array([0.0, 0.333, 1.0])

        assert len(unique_vals) == 3
        assert np.allclose(unique_vals, expected, atol=1e-3)

    def test_ordinal_feature_ranges(self):
        """测试 ordinal 特征范围计算"""
        X_train = np.array([
            [0.0, 10.0],
            [0.5, 15.0],
            [1.0, 20.0]
        ])

        # 计算特征范围
        mn = X_train.min(axis=0)
        mx = X_train.max(axis=0)

        # Ordinal 维（归一化后）应该在 [0, 1]
        assert mn[0] == pytest.approx(0.0)
        assert mx[0] == pytest.approx(1.0)

        # 连续维保持原始范围
        assert mn[1] == pytest.approx(10.0)
        assert mx[1] == pytest.approx(20.0)


class TestOrdinalNearestNeighbor:
    """最近邻约束测试"""

    def test_nearest_neighbor_constraint(self):
        """测试最近邻约束逻辑"""
        # 有效规范化值
        valid_values = np.array([0.0, 0.333, 1.0])

        # 扰动值（可能超出范围或不在有效值上）
        perturbed_values = np.array([0.1, 0.4, 0.8, -0.1, 1.2])

        # 最近邻约束
        constrained = np.zeros_like(perturbed_values)
        for i, pv in enumerate(perturbed_values):
            closest_idx = np.argmin(np.abs(valid_values - pv))
            constrained[i] = valid_values[closest_idx]

        # 验证约束后的值
        assert constrained[0] == pytest.approx(0.0)      # 0.1 → 0.0
        assert constrained[1] == pytest.approx(0.333)    # 0.4 → 0.333
        assert constrained[2] == pytest.approx(1.0)      # 0.8 → 1.0
        assert constrained[3] == pytest.approx(0.0)      # -0.1 → 0.0
        assert constrained[4] == pytest.approx(1.0)      # 1.2 → 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

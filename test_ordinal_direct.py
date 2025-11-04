"""
最小化测试 - 直接测试_check_ordinal_config方法
"""

import warnings
import torch
from unittest.mock import Mock

from eur_anova_pair import EURAnovaPairAcqf


def test_direct_check():
    """直接调用_check_ordinal_config方法"""
    print("\n=== 直接测试_check_ordinal_config ===")

    # 创建mock模型
    model = Mock()
    likelihood = Mock()

    # 设置类名
    type(likelihood).__name__ = "OrdinalLikelihood"
    likelihood.n_levels = 7

    # 确保没有cutpoints（Mock的getattr会返回Mock对象，我们需要明确不返回）
    def mock_getattr(name, default=None):
        if name in ["cutpoints", "cut_points", "cut_points_", "_cutpoints"]:
            return None
        return Mock()

    likelihood.__getattribute__ = lambda self, name: mock_getattr(name)

    model.likelihood = likelihood

    # 模拟训练数据
    train_inputs = (torch.tensor([[0.1, 0.2], [0.3, 0.4]]),)
    model.train_inputs = train_inputs

    print(f"模型设置:")
    print(f"  - likelihood类名: {type(likelihood).__name__}")
    print(f"  - n_levels: {getattr(likelihood, 'n_levels', None)}")

    # 创建采集函数对象（不触发数据同步）
    acqf = EURAnovaPairAcqf(
        model=model, variable_types={0: "continuous", 1: "continuous"}
    )

    # 捕获警告，直接调用检查方法
    print(f"\n直接调用_check_ordinal_config()...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        acqf._check_ordinal_config()

        print(f"\n警告数量: {len(w)}")
        if w:
            for i, warning in enumerate(w):
                print(f"\n警告内容:")
                print(f"{warning.message}")
                return True
        else:
            print("❌ 没有捕获到警告！")
            return False


if __name__ == "__main__":
    success = test_direct_check()
    if success:
        print("\n✅ 测试成功！警告正常触发")
    else:
        print("\n❌ 测试失败！警告未触发")

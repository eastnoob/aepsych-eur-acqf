"""
简化的调试测试 - 检查为什么警告没有触发
"""

import warnings
import torch
from unittest.mock import Mock

from eur_anova_pair import EURAnovaPairAcqf


def test_simple():
    """最简单的测试 - 添加调试输出"""
    print("\n=== 简化调试测试 ===")
    
    # 创建mock模型
    model = Mock()
    likelihood = Mock()
    
    # 设置类名
    type(likelihood).__name__ = "OrdinalLikelihood"
    likelihood.n_levels = 7
    
    # 确保没有cutpoints属性
    # Mock默认会返回Mock对象，我们需要明确设置为None或不存在
    def get_none(*args, **kwargs):
        return None
    
    likelihood.cutpoints = property(get_none)
    likelihood.cut_points = property(get_none)
    likelihood.cut_points_ = property(get_none)
    likelihood._cutpoints = property(get_none)
    
    model.likelihood = likelihood
    
    # 模拟训练数据
    train_inputs = (torch.tensor([[0.1, 0.2], [0.3, 0.4]]),)
    model.train_inputs = train_inputs
    
    print(f"模型设置完成:")
    print(f"  - likelihood类名: {type(likelihood).__name__}")
    print(f"  - 有n_levels: {hasattr(likelihood, 'n_levels')}")
    print(f"  - n_levels值: {getattr(likelihood, 'n_levels', None)}")
    
    # 测试cutpoints检查
    print(f"\n检查cutpoints属性:")
    for name in ["cutpoints", "cut_points", "cut_points_", "_cutpoints"]:
        val = getattr(likelihood, name, "NOT_FOUND")
        print(f"  - {name}: {val}")
    
    # 创建采集函数
    print(f"\n创建采集函数...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        acqf = EURAnovaPairAcqf(
            model=model,
            variable_types={0: "continuous", 1: "continuous"}
        )
        
        print(f"\n初始化后警告数量: {len(w)}")
        
        # 手动触发数据同步（这会调用_check_ordinal_config）
        print(f"\n手动触发_ensure_fresh_data()...")
        acqf._ensure_fresh_data()
        
        print(f"\n同步后警告数量: {len(w)}")
        if w:
            for i, warning in enumerate(w):
                print(f"警告 {i+1}: {warning.message}")
        else:
            print("没有捕获到警告！")
            
        # 检查标志
        print(f"\n检查标志:")
        print(f"  - 有_ordinal_check_done: {hasattr(acqf, '_ordinal_check_done')}")
        print(f"  - _ordinal_check_done值: {getattr(acqf, '_ordinal_check_done', 'NOT_SET')}")


if __name__ == "__main__":
    test_simple()

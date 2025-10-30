"""
测试 torch.topk 对 -inf 值的处理行为
"""

import torch
import numpy as np

# 测试1: topk 是否会自动跳过 -inf?
print("=" * 60)
print("测试1: torch.topk 是否会选择 -inf 值?")
print("=" * 60)

scores = torch.tensor([3.5, -np.inf, 7.2, -np.inf, 5.1, -np.inf, 8.9, 2.3])
print(f"原始scores: {scores}")
print(f"包含 -inf 的位置: {torch.isinf(scores).nonzero().flatten().tolist()}")

# 尝试选择 top-3
k = 3
topk_values, topk_indices = torch.topk(scores, k)
print(f"\ntorch.topk(scores, {k}):")
print(f"  Values: {topk_values}")
print(f"  Indices: {topk_indices}")

# 测试2: 如果 -inf 数量很多,会怎样?
print("\n" + "=" * 60)
print("测试2: 大量 -inf 时的行为")
print("=" * 60)

scores2 = torch.tensor([5.0, -np.inf, -np.inf, -np.inf, 3.0, -np.inf, 4.0, -np.inf])
print(f"原始scores: {scores2}")
print(f"非-inf值数量: {(~torch.isinf(scores2)).sum().item()}")

k = 5  # 请求5个,但只有4个非-inf
topk_values2, topk_indices2 = torch.topk(scores2, k)
print(f"\ntorch.topk(scores, {k}):")
print(f"  Values: {topk_values2}")
print(f"  Indices: {topk_indices2}")
print(f"  包含 -inf? {torch.isinf(topk_values2).any().item()}")

# 测试3: 当 k=1 且有 -inf 时?
print("\n" + "=" * 60)
print("测试3: k=1 时是否会选到 -inf")
print("=" * 60)

scores3 = torch.tensor([-np.inf, 5.0, -np.inf, 3.0])
print(f"原始scores: {scores3}")

k = 1
topk_values3, topk_indices3 = torch.topk(scores3, k)
print(f"\ntorch.topk(scores, {k}):")
print(f"  Values: {topk_values3}")
print(f"  Indices: {topk_indices3}")
print(f"  是否为 -inf? {torch.isinf(topk_values3).item()}")

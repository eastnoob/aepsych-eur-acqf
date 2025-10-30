# 修复记录 - 维度转换问题

## 问题

AEPsych的`AcqfGridSearchGenerator`在返回候选点时保留了q维度,导致参数转换时维度不匹配:

```
错误: IndexError: index 3 is out of bounds for dimension 0 with size 1
```

## 根本原因

在`grid_eval_acqf_generator.py`中:

```python
# Line 108-110
if len(X_rnd.shape) < 3:
    X_rnd = X_rnd.unsqueeze(1)  # 添加q维度: [samps, dim] → [samps, 1, dim]
```

然后在`acqf_grid_search_generator.py`中:

```python
# Line 44-48 (原始)
grid, acqf_vals = self._eval_acqf(...)  # grid shape: [1000, 1, 4]
_, idxs = torch.topk(acqf_vals, num_points)
new_candidate = grid[idxs]              # shape: [1, 1, 4]
return new_candidate                     # ❌ 错误shape!
```

Server期待`[1, 4]`但收到了`[1, 1, 4]`,当transforms尝试访问`X[..., 3]`时:

- `X[..., 3]`实际是`X[1, 3]` (把1,1当成batch和q维度)
- 但第0维只有size=1,所以`X[1, 3]`越界!

## 解决方案

在返回前squeeze掉q维度:

```python
# File: temp_aepsych/aepsych/generators/acqf_grid_search_generator.py
# Location: _gen() method, after line 48

new_candidate = grid[idxs]

# Remove q dimension if present (grid has shape [samps, 1, dim])
# Result should be [num_points, dim] for proper parameter transforms
if len(new_candidate.shape) == 3 and new_candidate.shape[1] == 1:
    new_candidate = new_candidate.squeeze(1)

return new_candidate  # ✅ 正确shape: [1, 4]
```

## 验证

修复后运行80个trial,**0个错误**:

```
[INFO] Starting gen...
[INFO] Gen done, time=0.5s    ← 成功!
```

Parameter transforms正确处理所有categorical和ordinal变量。

## 影响范围

只影响`AcqfGridSearchGenerator`,其他生成器不受影响:

- `OptimizeAcqfGenerator`: 使用`optimize_acqf`,返回正确shape
- `RandomGenerator`: 直接生成`[n, dim]`
- `ManualGenerator`: 接收用户输入

## 提交

**修改文件**: `temp_aepsych/aepsych/generators/acqf_grid_search_generator.py`

**修改位置**: `_gen()` method, lines 44-50

**修改类型**: Bug fix (shape mismatch)

**影响**: 修复q维度导致的参数转换错误

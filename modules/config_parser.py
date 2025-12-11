"""
配置解析模块

解析交互对、交互三元组和变量类型配置。
支持多种输入格式，自动去重和验证。

Example:
    >>> # 解析交互对
    >>> pairs = parse_interaction_pairs("0,1; 2,3")
    >>> # [(0, 1), (2, 3)]
    >>>
    >>> # 解析三阶交互
    >>> triplets = parse_interaction_triplets([(0,1,2), (1,2,3)])
    >>> # [(0, 1, 2), (1, 2, 3)]
    >>>
    >>> # 解析变量类型
    >>> vt = parse_variable_types("categorical, continuous, integer")
    >>> # {0: 'categorical', 1: 'continuous', 2: 'integer'}
"""

from __future__ import annotations
from typing import Union, Sequence, Tuple, List, Dict, Optional
import warnings
import numpy as np


def parse_interaction_pairs(
    interaction_pairs: Union[str, Sequence[Union[str, Tuple[int, int]]]],
) -> List[Tuple[int, int]]:
    """解析交互对输入

    支持格式：
    - [(0,1), (2,3)]           # 元组列表
    - "0,1; 2,3"              # 分号分隔
    - ["0,1", "2|3"]          # 混合分隔符

    关键特性：
    1. 自动去重（保持首次出现顺序）
    2. 自动规范化（i < j）
    3. 跳过自环（i == j）

    Args:
        interaction_pairs: 交互对配置

    Returns:
        去重后的交互对列表 [(i, j), ...]，其中 i < j

    Example:
        >>> parse_interaction_pairs("0,1; 2,3")
        [(0, 1), (2, 3)]
        >>> parse_interaction_pairs([(1,0), (0,1), (2,3)])
        [(0, 1), (2, 3)]  # 已去重
    """
    parsed = []
    seen = set()
    duplicate_count = 0

    # 统一转为列表
    seq = (
        [interaction_pairs]
        if isinstance(interaction_pairs, str)
        else list(interaction_pairs)
    )

    def _add_pair(i: int, j: int) -> None:
        """添加交互对（自动规范化和去重）"""
        nonlocal duplicate_count
        if i == j:
            return  # 跳过自环

        pair = (min(i, j), max(i, j))
        if pair not in seen:
            seen.add(pair)
            parsed.append(pair)
        else:
            duplicate_count += 1

    for it in seq:
        try:
            # 类型1：元组/列表
            if isinstance(it, (list, tuple)) and len(it) == 2:
                _add_pair(int(it[0]), int(it[1]))
                continue

            # 类型2：字符串
            s = str(it).strip().strip('"').strip("'")

            # 分割分隔符
            if ";" in s:
                pair_strs = s.split(";")
            elif " " in s and "," in s:
                pair_strs = s.split()
            else:
                pair_strs = [s]

            for ps in pair_strs:
                ps = ps.strip()
                if not ps:
                    continue

                # 解析单个对
                if "," in ps:
                    parts = ps.split(",")
                elif "|" in ps:
                    parts = ps.split("|")
                else:
                    warnings.warn(f"无法解析交互对格式: '{ps}' (需要包含 ',' 或 '|')")
                    continue

                if len(parts) >= 2:
                    try:
                        i = int(parts[0].strip().strip('"').strip("'"))
                        j = int(parts[1].strip().strip('"').strip("'"))
                        _add_pair(i, j)
                    except ValueError as e:
                        warnings.warn(f"无法解析交互对索引: '{ps}' (错误: {e})")

        except Exception as e:
            warnings.warn(f"解析交互对时出错: {it}, 错误: {e}")

    # 用户友好提示
    if duplicate_count > 0:
        warnings.warn(
            f"交互对输入包含 {duplicate_count} 个重复项，已自动去重（保持首次出现顺序）"
        )

    return parsed


def parse_interaction_triplets(
    interaction_triplets: Union[str, Sequence[Union[str, Tuple[int, int, int]]]],
) -> List[Tuple[int, int, int]]:
    """解析三阶交互三元组

    支持格式：
    - [(0,1,2), (1,2,3)]      # 元组列表
    - "0,1,2; 1,2,3"         # 分号分隔
    - ["0,1,2", "1|2|3"]     # 混合分隔符

    Args:
        interaction_triplets: 三元组配置

    Returns:
        去重后的三元组列表 [(i, j, k), ...]，其中 i < j < k

    Example:
        >>> parse_interaction_triplets("0,1,2; 1,2,3")
        [(0, 1, 2), (1, 2, 3)]
        >>> parse_interaction_triplets([(2,1,0), (0,1,2)])
        [(0, 1, 2)]  # 已去重和规范化
    """
    parsed = []
    seen = set()
    duplicate_count = 0

    # 统一转为列表
    seq = (
        [interaction_triplets]
        if isinstance(interaction_triplets, str)
        else list(interaction_triplets)
    )

    def _add_triplet(i: int, j: int, k: int) -> None:
        """添加三元组（自动规范化和去重）"""
        nonlocal duplicate_count

        # 跳过包含重复索引的三元组
        if len({i, j, k}) < 3:
            return

        triplet = tuple(sorted([i, j, k]))
        if triplet not in seen:
            seen.add(triplet)
            parsed.append(triplet)
        else:
            duplicate_count += 1

    for it in seq:
        try:
            # 类型1：元组/列表
            if isinstance(it, (list, tuple)) and len(it) == 3:
                _add_triplet(int(it[0]), int(it[1]), int(it[2]))
                continue

            # 类型2：字符串
            s = str(it).strip().strip('"').strip("'")

            # 分割分隔符
            if ";" in s:
                triplet_strs = s.split(";")
            elif " " in s and "," in s:
                triplet_strs = s.split()
            else:
                triplet_strs = [s]

            for ts in triplet_strs:
                ts = ts.strip()
                if not ts:
                    continue

                # 解析单个三元组
                if "," in ts:
                    parts = ts.split(",")
                elif "|" in ts:
                    parts = ts.split("|")
                else:
                    warnings.warn(f"无法解析三元组格式: '{ts}' (需要包含 ',' 或 '|')")
                    continue

                if len(parts) >= 3:
                    try:
                        i = int(parts[0].strip().strip('"').strip("'"))
                        j = int(parts[1].strip().strip('"').strip("'"))
                        k = int(parts[2].strip().strip('"').strip("'"))
                        _add_triplet(i, j, k)
                    except ValueError as e:
                        warnings.warn(f"无法解析三元组索引: '{ts}' (错误: {e})")

        except Exception as e:
            warnings.warn(f"解析三元组时出错: {it}, 错误: {e}")

    # 用户友好提示
    if duplicate_count > 0:
        warnings.warn(f"三元组输入包含 {duplicate_count} 个重复项，已自动去重")

    return parsed


def parse_variable_types(variable_types_list: Union[List[str], str]) -> Dict[int, str]:
    """解析变量类型配置

    支持格式：
    - "categorical, continuous, integer"       # 逗号分隔字符串
    - ["categorical", "continuous", "integer"] # 列表
    - "(cat; cont; int)"                       # 括号+分号

    识别规则：
    - 包含 'custom_ordinal_mono' → 'custom_ordinal_mono' (非等差有序)
    - 包含 'custom_ordinal' → 'custom_ordinal' (等差有序)
    - 以 'ord' 开头 → 'ordinal' (基础有序)
    - 以 'cat' 开头 → 'categorical'
    - 以 'int' 开头 → 'integer'
    - 以 'cont', 'float', 'real' 开头 → 'continuous'

    Args:
        variable_types_list: 变量类型配置

    Returns:
        {dim_idx: type_str}

    Example:
        >>> parse_variable_types("categorical, continuous, integer")
        {0: 'categorical', 1: 'continuous', 2: 'integer'}
        >>> parse_variable_types(["cat", "cont", "int"])
        {0: 'categorical', 1: 'continuous', 2: 'integer'}
        >>> parse_variable_types(["custom_ordinal", "continuous"])
        {0: 'custom_ordinal', 1: 'continuous'}
    """
    raw = variable_types_list

    if isinstance(raw, str):
        s = raw.strip().strip("[]()")
        parts = [p for p in s.replace(";", ",").split(",")]
    else:
        parts = list(raw)

    tokens: List[str] = []
    for p in parts:
        item = str(p).strip().strip('"').strip("'")
        if item:
            tokens.append(item)

    vt_map: Dict[int, str] = {}
    for i, t in enumerate(tokens):
        t_l = t.lower()
        # Ordinal types (most specific first)
        if "custom_ordinal_mono" in t_l:
            vt_map[i] = "custom_ordinal_mono"
        elif "custom_ordinal" in t_l:
            vt_map[i] = "custom_ordinal"
        elif t_l.startswith("ord"):
            vt_map[i] = "ordinal"
        # Other types
        elif t_l.startswith("cat"):
            vt_map[i] = "categorical"
        elif t_l.startswith("int"):
            vt_map[i] = "integer"
        elif (
            t_l.startswith("cont") or t_l.startswith("float") or t_l.startswith("real")
        ):
            vt_map.setdefault(i, "continuous")

    return vt_map


def parse_ard_weights(
    ard_weights: Union[str, Sequence[float], None],
) -> Optional[np.ndarray]:
    """解析ARD权重配置

    支持格式:
    - "0.1, 0.2, 0.3"           # 逗号分隔字符串
    - [0.1, 0.2, 0.3]           # 列表
    - "[0.1, 0.2, 0.3]"         # 带括号字符串
    - None                      # 不使用权重

    权重会自动归一化使得 sum(weights) = 1.0

    Args:
        ard_weights: ARD权重配置

    Returns:
        归一化后的权重数组，或None（不使用权重）
    """
    if ard_weights is None:
        return None

    # 字符串解析
    if isinstance(ard_weights, str):
        s = ard_weights.strip().strip("[]()")
        if not s:
            return None
        try:
            parts = [float(p.strip()) for p in s.split(",") if p.strip()]
        except ValueError as e:
            warnings.warn(f"无法解析ARD权重: '{ard_weights}' (错误: {e})")
            return None
    else:
        # 序列类型
        try:
            parts = [float(x) for x in ard_weights]
        except (TypeError, ValueError) as e:
            warnings.warn(f"无法解析ARD权重: {ard_weights} (错误: {e})")
            return None

    if len(parts) == 0:
        return None

    weights = np.array(parts, dtype=np.float64)

    # 验证: 全部非负
    if np.any(weights < 0):
        warnings.warn(f"ARD权重包含负值，已取绝对值: {weights}")
        weights = np.abs(weights)

    # 归一化
    total = weights.sum()
    if total < 1e-10:
        warnings.warn("ARD权重总和接近零，返回等权重")
        return np.ones(len(weights)) / len(weights)

    return weights / total


def validate_interaction_indices(
    pairs: List[Tuple[int, int]], triplets: List[Tuple[int, int, int]], n_dims: int
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """验证交互索引范围并过滤越界项

    Args:
        pairs: 二阶交互列表
        triplets: 三阶交互列表
        n_dims: 变量维度数

    Returns:
        (filtered_pairs, filtered_triplets)
    """
    # 过滤二阶
    invalid_pairs = [(i, j) for i, j in pairs if i >= n_dims or j >= n_dims]
    if invalid_pairs:
        warnings.warn(
            f"交互对包含越界索引（维度={n_dims}）：{invalid_pairs}，已自动过滤。"
        )
    valid_pairs = [(i, j) for i, j in pairs if i < n_dims and j < n_dims]

    # 过滤三阶
    invalid_triplets = [
        (i, j, k) for i, j, k in triplets if i >= n_dims or j >= n_dims or k >= n_dims
    ]
    if invalid_triplets:
        warnings.warn(
            f"三元组包含越界索引（维度={n_dims}）：{invalid_triplets}，已自动过滤。"
        )
    valid_triplets = [
        (i, j, k) for i, j, k in triplets if i < n_dims and j < n_dims and k < n_dims
    ]

    return valid_pairs, valid_triplets


def parse_diagnostics_options(options: Union[Dict[str, Any], None]) -> Dict[str, Any]:
    """Parse diagnostics/logging related options from a config-like mapping.

    Accepts a dict or None and returns normalized diagnostics settings.

    Returned keys:
      - diagnostics_enabled: bool
      - diagnostics_verbose: bool
      - diagnostics_output_file: Optional[str]
      - log_level: str
    """
    defaults = {
        "diagnostics_enabled": False,
        "diagnostics_verbose": False,
        "diagnostics_output_file": None,
        "log_level": "WARNING",
    }

    if not options:
        return defaults

    out = defaults.copy()
    # accept multiple key styles
    for k in out.keys():
        if k in options:
            out[k] = options[k]
        else:
            # accept shorter aliases
            alias = k.replace("diagnostics_", "")
            if alias in options:
                out[k] = options[alias]

    # coerce types
    out["diagnostics_enabled"] = str(out["diagnostics_enabled"]).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    out["diagnostics_verbose"] = str(out["diagnostics_verbose"]).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if out["diagnostics_output_file"] == "" or out["diagnostics_output_file"] is False:
        out["diagnostics_output_file"] = None
    out["log_level"] = str(out["log_level"]).upper()

    return out

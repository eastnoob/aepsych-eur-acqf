"""
V3 Acquisition Function: V1 + Hard Exclusion
采集函数V3：基于V1的最小化改进，仅添加硬重复排除

核心原则：
- 保持V1的全部框架（信息增益 + 覆盖度）
- 只添加硬排除机制消除重复采样
- 不引入新的组件或复杂度

预期改进：
- 唯一设计数: 39 → 55-65 (+40-65%)
- 高分发现: 8 → 12-16 (+50-100%)
"""

import numpy as np
from scipy.stats import norm
from configparser import ConfigParser
import sys
import os

# 添加父目录到路径以导入V1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acquisition_function import VarianceReductionWithCoverageAcqf


class HardExclusionAcqf(VarianceReductionWithCoverageAcqf):
    """
    V3方案A: V1 + 硬重复排除

    改进点：
    1. 硬排除已采样设计（score = -inf）
    2. 其他全部继承V1

    配置参数：
    - 与V1完全相同
    - 无需额外参数
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sampled_designs = set()  # 存储已采样设计的指纹

    def fit(self, X, y, variable_types=None):
        """拟合并记录采样历史"""
        super().fit(X, y, variable_types=variable_types)

        # 清空并重新记录所有已采样设计
        self._sampled_designs.clear()
        for x in X:
            design_key = self._design_to_key(x)
            self._sampled_designs.add(design_key)

        print(f"[HardExclusionAcqf] 已记录 {len(self._sampled_designs)} 个唯一设计")

    def _design_to_key(self, x):
        """将设计转换为唯一字符串标识"""
        if isinstance(x, np.ndarray):
            if x.ndim > 1:
                x = x.flatten()
            return "_".join([f"{float(val):.6f}" for val in x])
        return str(x)

    def _evaluate_numpy(self, X_candidates):
        """评估候选点，硬排除已采样设计"""
        # 使用V1的评分逻辑
        scores = super()._evaluate_numpy(X_candidates)

        # 硬排除已采样设计
        excluded_count = 0
        for i, x in enumerate(X_candidates):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                scores[i] = -np.inf  # 完全排除，而非软惩罚
                excluded_count += 1

        if excluded_count > 0:
            print(
                f"[HardExclusionAcqf] 本轮硬排除 {excluded_count}/{len(X_candidates)} 个已采样设计"
            )

        return scores

    def select_next(self, X_candidates, n_select=1):
        """
        选择下一个采样点，确保不重复
        
        关键改进：跳过已采样设计，选择未采样的top-n
        确保返回的数量 == n_select（即使需要选次优）
        """
        # 获取所有得分
        scores = self(X_candidates)
        
        # 标识未采样的设计
        unsampled_mask = np.ones(len(X_candidates), dtype=bool)
        for i, x in enumerate(X_candidates):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                unsampled_mask[i] = False
        
        # 获取未采样设计的索引
        unsampled_indices = np.where(unsampled_mask)[0]
        
        if len(unsampled_indices) == 0:
            # 如果所有候选都已采样（极端情况），随机选择
            print(f"[HardExclusionAcqf] 警告：所有候选设计都已采样，随机选择")
            selected_indices = np.random.choice(len(X_candidates), size=min(n_select, len(X_candidates)), replace=False)
        else:
            # 获取未采样设计的得分
            unsampled_scores = scores[unsampled_indices]
            
            # 按得分降序排列（使用argsort，然后反转）
            sorted_order = np.argsort(-unsampled_scores)  # 负号实现降序
            sorted_unsampled_idx = unsampled_indices[sorted_order]
            
            # 选择top-n
            n_available = min(n_select, len(sorted_unsampled_idx))
            selected_indices = sorted_unsampled_idx[:n_available]
            
            if n_available < n_select:
                print(f"[HardExclusionAcqf] 警告：仅有{n_available}个未采样设计，少于请求的{n_select}个")
        
        selected_X = X_candidates[selected_indices]
        
        return selected_X, selected_indices


class CombinedAcqf(VarianceReductionWithCoverageAcqf):
    """
    V3方案C: V1 + 候选集预过滤 + 硬排除

    改进点：
    1. 候选集生成时优先未采样设计（80%未采样 + 20%已采样）
    2. 评分时仍然硬排除已采样设计（双重保险）
    3. 其他全部继承V1

    配置参数：
    - 与V1完全相同
    - candidate_unsampled_ratio: 候选集中未采样设计的比例（默认0.8）
    """

    def __init__(self, candidate_unsampled_ratio=0.8, **kwargs):
        super().__init__(**kwargs)
        self._sampled_designs = set()
        self.candidate_unsampled_ratio = candidate_unsampled_ratio

    def fit(self, X, y, variable_types=None):
        """拟合并记录采样历史"""
        super().fit(X, y, variable_types=variable_types)

        # 清空并重新记录所有已采样设计
        self._sampled_designs.clear()
        for x in X:
            design_key = self._design_to_key(x)
            self._sampled_designs.add(design_key)

        print(f"[CombinedAcqf] 已记录 {len(self._sampled_designs)} 个唯一设计")

    def _design_to_key(self, x):
        """将设计转换为唯一字符串标识"""
        if isinstance(x, np.ndarray):
            if x.ndim > 1:
                x = x.flatten()
            return "_".join([f"{float(val):.6f}" for val in x])
        return str(x)

    def filter_candidates(self, X_candidates):
        """
        过滤候选集，优先未采样设计

        Args:
            X_candidates: 所有候选设计 (n_candidates, n_features)

        Returns:
            filtered: 过滤后的候选集，优先包含未采样设计
        """
        if len(self._sampled_designs) == 0:
            return X_candidates

        # 分类候选集
        unsampled_idx = []
        sampled_idx = []

        for i, x in enumerate(X_candidates):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                sampled_idx.append(i)
            else:
                unsampled_idx.append(i)

        # 计算目标数量
        n_total = len(X_candidates)
        n_unsampled_target = int(n_total * self.candidate_unsampled_ratio)
        n_sampled_target = n_total - n_unsampled_target

        # 采样
        n_unsampled = min(len(unsampled_idx), n_unsampled_target)
        n_sampled = min(len(sampled_idx), n_sampled_target)

        # 如果未采样设计不足，用已采样设计补充
        if n_unsampled < n_unsampled_target:
            n_sampled = min(len(sampled_idx), n_total - n_unsampled)

        selected_idx = []
        if n_unsampled > 0:
            selected_idx.extend(
                np.random.choice(unsampled_idx, size=n_unsampled, replace=False)
            )
        if n_sampled > 0:
            selected_idx.extend(
                np.random.choice(sampled_idx, size=n_sampled, replace=False)
            )

        return X_candidates[selected_idx]

    def _evaluate_numpy(self, X_candidates):
        """评估候选点，硬排除已采样设计"""
        # 使用V1的评分逻辑
        scores = super()._evaluate_numpy(X_candidates)

        # 硬排除已采样设计（双重保险）
        for i, x in enumerate(X_candidates):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                scores[i] = -np.inf

        return scores

    def select_next(self, X_candidates, n_select=1):
        """
        选择下一个采样点，确保不重复
        
        关键改进：跳过已采样设计，选择未采样的top-n
        确保返回的数量 == n_select（即使需要选次优）
        """
        # 获取所有得分
        scores = self(X_candidates)
        
        # 标识未采样的设计
        unsampled_mask = np.ones(len(X_candidates), dtype=bool)
        for i, x in enumerate(X_candidates):
            design_key = self._design_to_key(x)
            if design_key in self._sampled_designs:
                unsampled_mask[i] = False
        
        # 获取未采样设计的索引
        unsampled_indices = np.where(unsampled_mask)[0]
        
        if len(unsampled_indices) == 0:
            # 如果所有候选都已采样（极端情况），随机选择
            print(f"[CombinedAcqf] 警告：所有候选设计都已采样，随机选择")
            selected_indices = np.random.choice(len(X_candidates), size=min(n_select, len(X_candidates)), replace=False)
        else:
            # 获取未采样设计的得分
            unsampled_scores = scores[unsampled_indices]
            
            # 按得分降序排列（使用argsort，然后反转）
            sorted_order = np.argsort(-unsampled_scores)  # 负号实现降序
            sorted_unsampled_idx = unsampled_indices[sorted_order]
            
            # 选择top-n
            n_available = min(n_select, len(sorted_unsampled_idx))
            selected_indices = sorted_unsampled_idx[:n_available]
            
            if n_available < n_select:
                print(f"[CombinedAcqf] 警告：仅有{n_available}个未采样设计，少于请求的{n_select}个")
        
        selected_X = X_candidates[selected_indices]
        
        return selected_X, selected_indices


# 配置加载支持
def load_from_config(config_str, acqf_type="hard_exclusion"):
    """
    从配置字符串加载V3采集函数

    Args:
        config_str: INI格式配置字符串
        acqf_type: "hard_exclusion" 或 "combined"

    Returns:
        采集函数实例
    """
    config = ConfigParser()
    config.read_string(config_str)

    # V3使用与V1相同的配置section
    section = "VarianceReductionWithCoverageAcqf"
    if not config.has_section(section):
        section = list(config.sections())[-1]  # 使用最后一个section

    # 解析参数 (使用V1的参数名)
    params = {}

    # 基础参数
    if config.has_option(section, "lambda_min"):
        params["lambda_min"] = config.getfloat(section, "lambda_min")
    if config.has_option(section, "lambda_max"):
        params["lambda_max"] = config.getfloat(section, "lambda_max")
    if config.has_option(section, "tau_1"):
        params["tau_1"] = config.getfloat(section, "tau_1")
    if config.has_option(section, "tau_2"):
        params["tau_2"] = config.getfloat(section, "tau_2")
    if config.has_option(section, "gamma"):
        params["gamma"] = config.getfloat(section, "gamma")

    # 覆盖度方法
    if config.has_option(section, "coverage_method"):
        params["coverage_method"] = config.get(section, "coverage_method")

    # 交互项
    if config.has_option(section, "interaction_terms"):
        interaction_str = config.get(section, "interaction_terms")
        params["interaction_terms"] = []
        for term in interaction_str.split(";"):
            term = term.strip()
            if term:
                indices = term.strip("()").split(",")
                params["interaction_terms"].append(
                    (int(indices[0].strip()), int(indices[1].strip()))
                )

    # 组合方案的额外参数
    if acqf_type == "combined":
        if config.has_option(section, "candidate_unsampled_ratio"):
            params["candidate_unsampled_ratio"] = config.getfloat(
                section, "candidate_unsampled_ratio"
            )

    # 创建实例
    if acqf_type == "hard_exclusion":
        return HardExclusionAcqf(**params)
    elif acqf_type == "combined":
        return CombinedAcqf(**params)
    else:
        raise ValueError(f"Unknown acqf_type: {acqf_type}")


if __name__ == "__main__":
    # 简单测试
    print("V3 Acquisition Functions - 测试不重复选择逻辑")
    print("=" * 70)

    # 测试硬排除
    print("\n方案A: 硬排除")
    # V1参数名: lambda_min, lambda_max, tau_1, tau_2, gamma
    acqf_a = HardExclusionAcqf(
        model=None,  # 测试时不需要实际模型
        lambda_min=0.5,
        lambda_max=3.0,
        tau_1=0.5,
        tau_2=0.3,
        gamma=0.5,
        interaction_terms=[(0, 1), (1, 2)],
    )

    # 模拟数据
    X_train = np.array([[0, 1, 2], [1, 0, 1], [2, 2, 0]])
    y_train = np.array([0.8, 0.6, 0.7])
    acqf_a.fit(X_train, y_train)

    print(f"已采样设计: {acqf_a._sampled_designs}")

    # 测试1: select_next是否跳过已采样设计
    print("\n【测试1: select_next跳过已采样设计】")
    X_candidates = np.array([
        [0, 1, 2],  # 已采样 (应该跳过)
        [1, 0, 1],  # 已采样 (应该跳过)
        [2, 2, 0],  # 已采样 (应该跳过)
        [3, 3, 3],  # 未采样
        [4, 4, 4],  # 未采样
        [5, 5, 5],  # 未采样
    ])
    
    selected_X, selected_idx = acqf_a.select_next(X_candidates, n_select=2)
    print(f"请求选择: 2个设计")
    print(f"选中的索引: {selected_idx}")
    print(f"选中的设计: {selected_X.tolist()}")
    
    # 验证是否未采样
    all_unsampled = True
    for x in selected_X:
        key = acqf_a._design_to_key(x)
        if key in acqf_a._sampled_designs:
            all_unsampled = False
            print(f"❌ 错误：选中了已采样设计 {x.tolist()}")
    
    if all_unsampled:
        print(f"✅ 正确：所有选中的设计都是未采样的")
    
    # 测试2: 当候选集中有很多已采样设计时
    print("\n【测试2: 候选集中80%已采样】")
    X_many_sampled = np.array([
        [0, 1, 2], [1, 0, 1], [2, 2, 0],  # 前3个已采样
        [0, 1, 2], [1, 0, 1], [2, 2, 0],  # 重复的已采样
        [0, 1, 2], [1, 0, 1],              # 又是已采样
        [3, 3, 3], [4, 4, 4]               # 最后2个未采样
    ])
    
    selected_X2, selected_idx2 = acqf_a.select_next(X_many_sampled, n_select=2)
    print(f"候选集: 10个设计，其中8个已采样，2个未采样")
    print(f"选中的索引: {selected_idx2}")
    print(f"选中的设计: {selected_X2.tolist()}")
    
    # 验证
    all_unsampled2 = True
    for x in selected_X2:
        key = acqf_a._design_to_key(x)
        if key in acqf_a._sampled_designs:
            all_unsampled2 = False
            print(f"❌ 错误：选中了已采样设计 {x.tolist()}")
    
    if all_unsampled2:
        print(f"✅ 正确：即使候选集80%已采样，仍然只选择了未采样设计")
    
    # 测试组合方案
    print("\n" + "="*70)
    print("方案C: 组合方案")
    acqf_c = CombinedAcqf(
        model=None,
        lambda_min=0.5,
        lambda_max=3.0,
        tau_1=0.5,
        tau_2=0.3,
        gamma=0.5,
        candidate_unsampled_ratio=0.8,
        interaction_terms=[(0, 1), (1, 2)],
    )
    acqf_c.fit(X_train, y_train)

    # 测试候选集过滤 + select_next
    print("\n【测试3: 组合方案的候选集过滤】")
    filtered = acqf_c.filter_candidates(X_many_sampled)
    print(f"原始候选集大小: {len(X_many_sampled)}")
    print(f"过滤后候选集大小: {len(filtered)}")
    
    selected_X3, selected_idx3 = acqf_c.select_next(X_many_sampled, n_select=2)
    print(f"选中的设计: {selected_X3.tolist()}")
    
    # 验证
    all_unsampled3 = True
    for x in selected_X3:
        key = acqf_c._design_to_key(x)
        if key in acqf_c._sampled_designs:
            all_unsampled3 = False
            print(f"❌ 错误：选中了已采样设计 {x.tolist()}")
    
    if all_unsampled3:
        print(f"✅ 正确：组合方案也正确跳过已采样设计")

    print("\n" + "="*70)
    print("✅ V3不重复选择逻辑测试完成")
    print("="*70)

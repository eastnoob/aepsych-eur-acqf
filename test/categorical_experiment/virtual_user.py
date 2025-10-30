"""
虚拟用户模拟器 - 用于分类/有序离散变量实验

模拟用户对界面设计的偏好评分
"""

import numpy as np
from typing import Dict, List, Optional


class VirtualUser:
    """
    虚拟用户模拟器 - 评价UI设计方案

    设计变量:
    - color_scheme: 色彩方案 (5种: blue, green, red, purple, orange)
    - layout: 布局类型 (4种: grid, list, card, timeline)
    - font_size: 字体大小 (6级: 12, 14, 16, 18, 20, 22)
    - animation: 动画效果 (3种: none, subtle, dynamic)

    设计空间: 5×4×6×3 = 360种组合

    评分: 1-10的有序离散值
    """

    # 变量定义
    COLOR_SCHEMES = ["blue", "green", "red", "purple", "orange"]
    LAYOUTS = ["grid", "list", "card", "timeline"]
    FONT_SIZES = [12, 14, 16, 18, 20, 22]
    ANIMATIONS = ["none", "subtle", "dynamic"]

    # 评分范围
    MIN_RATING = 1
    MAX_RATING = 10

    def __init__(
        self,
        user_type: str = "balanced",
        noise_level: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        初始化虚拟用户

        Parameters
        ----------
        user_type : str
            用户类型,影响偏好模式:
            - "balanced": 平衡型(喜欢中等设置)
            - "minimalist": 极简型(喜欢简洁设计)
            - "colorful": 多彩型(喜欢丰富视觉)
        noise_level : float
            观测噪声水平(0-1),模拟用户评分的不一致性
        seed : int, optional
            随机种子
        """
        self.user_type = user_type
        self.noise_level = noise_level
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # 试次记录
        self.n_trials = 0
        self.trial_history = []

        # 根据用户类型设置偏好参数
        self._setup_preferences()

    def _setup_preferences(self):
        """设置用户偏好参数"""
        if self.user_type == "balanced":
            # 平衡型: 喜欢蓝绿色、卡片/网格布局、中等字体、微妙动画
            self.color_prefs = {
                "blue": 3.0,
                "green": 2.8,
                "purple": 2.0,
                "orange": 1.5,
                "red": 1.0,
            }
            self.layout_prefs = {"card": 2.5, "grid": 2.3, "list": 1.8, "timeline": 1.5}
            # 字体大小偏好: 16最佳,距离越远越差
            self.optimal_font = 16
            # 动画偏好
            self.animation_prefs = {"subtle": 2.0, "none": 1.5, "dynamic": 1.0}

        elif self.user_type == "minimalist":
            # 极简型: 喜欢蓝色、列表布局、小字体、无动画
            self.color_prefs = {
                "blue": 3.0,
                "green": 2.0,
                "purple": 1.5,
                "orange": 1.0,
                "red": 0.8,
            }
            self.layout_prefs = {"list": 3.0, "grid": 2.0, "card": 1.5, "timeline": 1.0}
            self.optimal_font = 14
            self.animation_prefs = {"none": 3.0, "subtle": 1.5, "dynamic": 0.5}

        elif self.user_type == "colorful":
            # 多彩型: 喜欢鲜艳颜色、卡片/时间线、大字体、动态动画
            self.color_prefs = {
                "red": 3.0,
                "orange": 2.8,
                "purple": 2.5,
                "green": 2.0,
                "blue": 1.5,
            }
            self.layout_prefs = {"card": 3.0, "timeline": 2.5, "grid": 1.8, "list": 1.0}
            self.optimal_font = 18
            self.animation_prefs = {"dynamic": 3.0, "subtle": 2.0, "none": 0.5}
        else:
            raise ValueError(f"Unknown user type: {self.user_type}")

    def _true_preference(self, design: Dict) -> float:
        """
        计算用户对设计的真实偏好分数(内部函数,实验中不可见)

        Parameters
        ----------
        design : dict
            设计参数 {'color_scheme': ..., 'layout': ..., 'font_size': ..., 'animation': ...}

        Returns
        -------
        float
            真实偏好分数(未归一化)
        """
        score = 0.0

        # 1. 色彩方案贡献
        color = design["color_scheme"]
        score += self.color_prefs.get(color, 1.0)

        # 2. 布局贡献
        layout = design["layout"]
        score += self.layout_prefs.get(layout, 1.0)

        # 3. 字体大小贡献(距离最优值越近越好)
        font_size = design["font_size"]
        font_diff = abs(font_size - self.optimal_font)
        # 使用高斯型衰减
        font_score = 2.5 * np.exp(-0.02 * font_diff**2)
        score += font_score

        # 4. 动画贡献
        animation = design["animation"]
        score += self.animation_prefs.get(animation, 1.0)

        # 5. 交互效应
        # 5.1 色彩-布局交互
        if (color in ["red", "orange"] and layout == "card") or (
            color == "blue" and layout == "list"
        ):
            score += 1.2  # 协调组合

        # 5.2 字体-动画交互
        if font_size >= 18 and animation == "dynamic":
            score += 0.8  # 大字体配动态动画更突出
        elif font_size <= 14 and animation == "none":
            score += 0.6  # 小字体配无动画更简洁

        # 5.3 布局-动画交互
        if layout == "timeline" and animation == "subtle":
            score += 1.0  # 时间线配微妙动画效果好

        return score

    def rate_design(self, design: Dict) -> Dict:
        """
        用户评价设计方案

        Parameters
        ----------
        design : dict
            设计参数

        Returns
        -------
        dict
            包含rating(评分)和其他元数据
        """
        # 计算真实偏好分数
        true_score = self._true_preference(design)

        # 添加噪声
        noise = np.random.normal(0, self.noise_level * 2)
        noisy_score = true_score + noise

        # 转换到1-10离散评分
        # 映射: score范围约为[4, 12] -> [1, 10]
        normalized = (noisy_score - 4) / 8 * 9 + 1
        rating = int(np.clip(np.round(normalized), self.MIN_RATING, self.MAX_RATING))

        # 模拟响应时间(评分越极端,决策越快)
        extremeness = abs(rating - 5.5) / 4.5
        rt = 1.0 + np.random.exponential(0.5) * (1 - extremeness)

        # 记录试次
        trial_data = {
            "trial": self.n_trials,
            "design": design.copy(),
            "rating": rating,
            "true_score": true_score,
            "rt": rt,
        }
        self.trial_history.append(trial_data)
        self.n_trials += 1

        return {"rating": rating, "rt": rt, "trial": self.n_trials - 1}

    def get_ground_truth(self, design: Dict) -> float:
        """
        获取设计的真实偏好分数(仅用于评估)

        Parameters
        ----------
        design : dict
            设计参数

        Returns
        -------
        float
            真实偏好分数
        """
        return self._true_preference(design)

    def get_all_designs(self) -> List[Dict]:
        """
        获取所有可能的设计组合

        Returns
        -------
        list
            所有设计组合的列表
        """
        designs = []
        for color in self.COLOR_SCHEMES:
            for layout in self.LAYOUTS:
                for font_size in self.FONT_SIZES:
                    for animation in self.ANIMATIONS:
                        designs.append(
                            {
                                "color_scheme": color,
                                "layout": layout,
                                "font_size": font_size,
                                "animation": animation,
                            }
                        )
        return designs

    @staticmethod
    def design_space_size() -> int:
        """返回设计空间大小"""
        return (
            len(VirtualUser.COLOR_SCHEMES)
            * len(VirtualUser.LAYOUTS)
            * len(VirtualUser.FONT_SIZES)
            * len(VirtualUser.ANIMATIONS)
        )


def test_virtual_user():
    """测试虚拟用户"""
    print("=" * 80)
    print(" 虚拟用户测试")
    print("=" * 80)

    user = VirtualUser(user_type="balanced", noise_level=0.5, seed=42)

    print(f"\n设计空间大小: {VirtualUser.design_space_size()}")
    print(f"1/4采样限制: {VirtualUser.design_space_size() // 4}")

    # 测试几个设计
    test_designs = [
        {
            "color_scheme": "blue",
            "layout": "card",
            "font_size": 16,
            "animation": "subtle",
        },
        {
            "color_scheme": "red",
            "layout": "list",
            "font_size": 12,
            "animation": "dynamic",
        },
        {
            "color_scheme": "green",
            "layout": "grid",
            "font_size": 20,
            "animation": "none",
        },
    ]

    print("\n测试设计评分:")
    for i, design in enumerate(test_designs, 1):
        result = user.rate_design(design)
        true_score = user.get_ground_truth(design)
        print(f"\n设计 {i}:")
        print(f"  {design}")
        print(f"  真实分数: {true_score:.2f}")
        print(f"  用户评分: {result['rating']}/10")
        print(f"  响应时间: {result['rt']:.2f}s")


if __name__ == "__main__":
    test_virtual_user()

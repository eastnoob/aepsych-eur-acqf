# -*- coding: utf-8 -*-
"""
虚拟被试模拟器

模拟一个真实被试对刺激参数的响应。
被试的响应基于一个潜在的心理物理函数,加上观测噪声。
"""

import numpy as np
from typing import Dict, Any


class VirtualSubject:
    """
    虚拟被试类

    模拟被试根据刺激参数给出响应的过程。
    内部使用一个"真实"函数来生成响应,但对外界不可见。
    """

    def __init__(
        self,
        true_function_type: str = "nonlinear_interaction",
        noise_std: float = 0.3,
        response_type: str = "continuous",
        seed: int = None,
    ):
        """
        初始化虚拟被试

        Parameters
        ----------
        true_function_type : str
            真实函数类型:
            - "linear": 纯线性函数
            - "nonlinear": 非线性主效应
            - "interaction": 线性+交互效应
            - "nonlinear_interaction": 非线性+交互效应(默认,最接近现实)
        noise_std : float
            观测噪声标准差
        response_type : str
            响应类型: "continuous"(连续) 或 "binary"(二值)
        seed : int, optional
            随机种子
        """
        self.true_function_type = true_function_type
        self.noise_std = noise_std
        self.response_type = response_type

        if seed is not None:
            np.random.seed(seed)

        # 记录所有试次
        self.trial_history = []
        self.n_trials = 0

        print(f"虚拟被试已创建:")
        print(f"  - 真实函数: {true_function_type}")
        print(f"  - 噪声水平: σ={noise_std}")
        print(f"  - 响应类型: {response_type}")

    def _true_function(self, x: np.ndarray) -> float:
        """
        被试的真实心理物理函数(对外界不可见)

        模拟真实的心理物理响应:
        - x1: 刺激强度(如亮度、音量)
        - x2: 刺激持续时间
        - x3: 背景噪声水平

        返回被试的"真实"感知强度(连续值)或判断(二值)
        """
        x1, x2, x3 = x[0], x[1], x[2]

        if self.true_function_type == "linear":
            # 纯线性: y = 2*x1 + 3*x2 + 1.5*x3
            y = 2.0 * x1 + 3.0 * x2 + 1.5 * x3

        elif self.true_function_type == "nonlinear":
            # 非线性主效应: 使用对数和平方根(模拟韦伯定律等)
            y = 3.0 * np.sqrt(x1) + 2.0 * np.log1p(x2 * 5) + 1.5 * x3**0.7

        elif self.true_function_type == "interaction":
            # 线性+交互: 刺激强度和持续时间的交互效应
            y = 2.0 * x1 + 3.0 * x2 + 1.5 * x3
            y += 1.8 * x1 * x2  # 强度×持续时间
            y += 1.2 * x2 * x3  # 持续时间×噪声

        elif self.true_function_type == "nonlinear_interaction":
            # 非线性+交互(最接近真实心理物理函数)
            # 主效应: 非线性响应
            y = 3.5 * np.sqrt(x1) + 2.5 * np.log1p(x2 * 5) + 1.8 * x3**0.7

            # 交互效应: 高强度+长持续时间=更强感知
            y += 2.0 * (x1**0.5) * (x2**0.6)  # 强度×持续时间(非线性)

            # 噪声抑制效应: 背景噪声降低感知
            y -= 0.8 * x3 * (x1 + x2) / 2  # 噪声×平均刺激强度

        else:
            raise ValueError(f"Unknown function type: {self.true_function_type}")

        return y

    def respond(self, stimulus: Dict[str, float]) -> Dict[str, Any]:
        """
        被试对刺激做出响应

        这是对外的接口,模拟被试看到刺激后给出响应的过程。

        Parameters
        ----------
        stimulus : Dict[str, float]
            刺激参数,如 {"x1": 0.5, "x2": 0.3, "x3": 0.7}

        Returns
        -------
        Dict[str, Any]
            响应字典,包含:
            - "response": 响应值(连续或二值)
            - "stimulus": 刺激参数
            - "trial": 试次编号
            - "rt": 反应时(模拟,可选)
        """
        # 将字典转为数组
        x = np.array([stimulus["x1"], stimulus["x2"], stimulus["x3"]])

        # 计算真实函数值
        true_value = self._true_function(x)

        # 添加观测噪声
        noisy_value = true_value + np.random.normal(0, self.noise_std)

        # 根据响应类型处理
        if self.response_type == "continuous":
            response = noisy_value
        elif self.response_type == "binary":
            # 使用sigmoid转换为概率,然后采样
            threshold = 5.0  # 判断阈值
            prob = 1 / (1 + np.exp(-(noisy_value - threshold)))
            response = 1 if np.random.rand() < prob else 0
        else:
            raise ValueError(f"Unknown response type: {self.response_type}")

        # 模拟反应时(可选,增加真实感)
        # 更困难的判断(接近阈值)需要更长时间
        base_rt = 0.5  # 基础反应时(秒)
        difficulty = abs(noisy_value - 5.0) / 5.0  # 困难度
        rt = base_rt + np.random.exponential(0.3) * (1 - difficulty)

        # 记录试次
        trial_data = {
            "trial": self.n_trials,
            "stimulus": stimulus.copy(),
            "response": response,
            "true_value": true_value,  # 仅用于分析,实验中不可见
            "rt": rt,
        }
        self.trial_history.append(trial_data)
        self.n_trials += 1

        # 返回响应数据(模拟实验中包含true_value用于分析)
        return {
            "response": response,
            "stimulus": stimulus,
            "trial": self.n_trials - 1,
            "rt": rt,
            "true_value": true_value,  # 模拟实验中可用于分析
        }

    def get_history(self):
        """获取所有试次历史"""
        return self.trial_history

    def get_ground_truth_at(self, x: np.ndarray) -> float:
        """
        获取某点的真实函数值(仅用于评估,实验中不可用)

        Parameters
        ----------
        x : np.ndarray
            刺激参数 [x1, x2, x3]

        Returns
        -------
        float
            真实函数值(无噪声)
        """
        return self._true_function(x)


class SubjectPool:
    """
    被试池 - 管理多个虚拟被试

    可以模拟被试间变异性
    """

    def __init__(
        self,
        n_subjects: int = 1,
        function_type: str = "nonlinear_interaction",
        noise_range: tuple = (0.2, 0.4),
        seed: int = 42,
    ):
        """
        创建被试池

        Parameters
        ----------
        n_subjects : int
            被试数量
        function_type : str
            真实函数类型
        noise_range : tuple
            噪声水平范围(最小,最大)
        seed : int
            随机种子
        """
        np.random.seed(seed)

        self.subjects = []
        for i in range(n_subjects):
            # 每个被试有略微不同的噪声水平
            noise = np.random.uniform(noise_range[0], noise_range[1])
            subject = VirtualSubject(
                true_function_type=function_type, noise_std=noise, seed=seed + i
            )
            self.subjects.append(subject)

        self.current_subject_idx = 0
        print(f"\n被试池已创建: {n_subjects}名被试")

    def get_current_subject(self) -> VirtualSubject:
        """获取当前被试"""
        return self.subjects[self.current_subject_idx]

    def next_subject(self):
        """切换到下一名被试"""
        self.current_subject_idx = (self.current_subject_idx + 1) % len(self.subjects)


if __name__ == "__main__":
    # 测试虚拟被试
    print("=" * 70)
    print("虚拟被试测试")
    print("=" * 70)

    # 创建被试
    subject = VirtualSubject(
        true_function_type="nonlinear_interaction",
        noise_std=0.3,
        response_type="continuous",
        seed=42,
    )

    # 模拟几次试次
    print("\n模拟试次:")
    for i in range(5):
        stimulus = {
            "x1": np.random.rand(),
            "x2": np.random.rand(),
            "x3": np.random.rand(),
        }
        response = subject.respond(stimulus)
        print(
            f"  试次 {i}: 刺激={[f'{v:.2f}' for v in stimulus.values()]}, "
            f"响应={response['response']:.3f}, RT={response['rt']:.3f}s"
        )

    print("\n✓ 虚拟被试测试完成!")

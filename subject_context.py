"""
全局subject context - 用于追踪当前正在处理的subject_id

在MultiSubjectInheritanceRun中创建subject时设置此context，
LocalSampler初始化时读取此context来获得可复现的但不同的seed。
"""

import threading
from typing import Optional

_context_lock = threading.Lock()
_current_subject_id: Optional[int] = None


def set_current_subject_id(subject_id: Optional[int]) -> None:
    """设置当前正在处理的subject_id
    
    Args:
        subject_id: Subject编号（1-based），如果为None则清除设置
    """
    global _current_subject_id
    with _context_lock:
        _current_subject_id = subject_id


def get_current_subject_id() -> Optional[int]:
    """获取当前正在处理的subject_id
    
    Returns:
        当前subject_id，或None如果未设置
    """
    with _context_lock:
        return _current_subject_id


def reset_subject_context() -> None:
    """重置subject context"""
    global _current_subject_id
    with _context_lock:
        _current_subject_id = None

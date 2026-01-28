"""
因子模块
提供因子定义、因子计算、因子分析和评估功能
"""

from .base_factor import BaseFactor
from .factor_metrics import FactorMetrics
from .multi_factor_analysis import MultiFactorAnalysis
from .alpha158 import Alpha158

__all__ = [
    'BaseFactor',
    'FactorMetrics',
    'MultiFactorAnalysis',
    'Alpha158',
]

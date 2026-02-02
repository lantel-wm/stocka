"""
A股量化回测框架
一个简洁、准确且可扩展的量化回测框架
"""

__version__ = "1.0.0"
__author__ = "Stocka Team"

from .data.data_handler import DataHandler
from .strategy.base_strategy import BaseStrategy, Signal
from .strategy.ma_strategy import SimpleMAStrategy, DoubleMAStrategy, MultiMAStrategy
from .strategy.ml_strategy import MLStrategy
from .portfolio.portfolio import Portfolio
from .execution.transaction_cost import TransactionCost, StandardCost
from .backtest.engine import BacktestEngine
from .performance.analyzer import Performance, calculate_all_metrics
from .realtime.data_updater import DataUpdater
from .realtime.live_trader import LiveTrader
from .utils.config import Config, load_config
from .factor import (
    BaseFactor,
    FactorMetrics,
    MultiFactorAnalysis,
    Alpha158
)
from .model import LGBModel
from .pipeline import MLPipeline

__all__ = [
    'DataHandler',
    'BaseStrategy',
    'Signal',
    'SimpleMAStrategy',
    'DoubleMAStrategy',
    'MultiMAStrategy',
    'MLStrategy',
    'Portfolio',
    'TransactionCost',
    'StandardCost',
    'BacktestEngine',
    'Performance',
    'calculate_all_metrics',
    'DataUpdater',
    'LiveTrader',
    'Config',
    'load_config',
    'BaseFactor',
    'FactorMetrics',
    'MultiFactorAnalysis',
    'Alpha158',
    'LGBModel',
    'MLPipeline',
]


"""
策略基类模块
定义所有策略的基础接口和数据结构
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd


class Signal:
    """
    交易信号类
    表示单个股票的买卖信号
    """
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

    def __init__(self):
        self.date: Optional[date] = None
        self.code: str = ""
        self.signal_type: str = ""  # buy/sell/hold
        self.price: float = 0.0  # 建议价格
        self.weight: float = 1.0  # 仓位权重（0-1）
        self.reason: str = ""  # 信号原因

    def __repr__(self) -> str:
        return (f"Signal(date={self.date}, code={self.code}, "
                f"type={self.signal_type}, price={self.price:.2f}, "
                f"weight={self.weight:.2f}, reason={self.reason})")

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'date': self.date.strftime('%Y-%m-%d') if self.date else None,
            'code': self.code,
            'signal_type': self.signal_type,
            'price': self.price,
            'weight': self.weight,
            'reason': self.reason
        }


class BaseStrategy(ABC):
    """
    策略基类
    所有交易策略都应继承此类并实现 on_bar 方法
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        初始化策略

        Args:
            params: 策略参数字典
        """
        self.params = params or {}
        self.name = self.__class__.__name__
        self.indicators = {}  # 缓存计算好的指标
        self.min_bars = self.params.get('min_bars', 0)  # 策略需要的最少bar数量
        self.stock_list = self.params.get('stock_list', None)  # 策略股票列表（可选）

    def get_target_codes(self, data_handler) -> List[str]:
        """
        获取策略应该使用的股票代码列表

        Args:
            data_handler: 数据处理器对象

        Returns:
            股票代码列表
        """
        # 如果策略指定了股票列表，使用策略的列表
        if self.stock_list is not None:
            # 确保返回字符串列表
            return [str(code) for code in self.stock_list]

        # 否则使用 DataHandler 中的所有股票
        return data_handler.get_all_codes()

    @abstractmethod
    def on_bar(self, data_handler, current_date: date, portfolio) -> List[Signal]:
        """
        每日运行策略逻辑（抽象方法，必须由子类实现）

        Args:
            data_handler: 数据处理器对象
            current_date: 当前交易日期
            portfolio: 投资组合对象

        Returns:
            交易信号列表
        """
        raise NotImplementedError("子类必须实现 on_bar 方法")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标（可选方法）

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            添加了技术指标的DataFrame
        """
        # 默认不计算任何指标，子类可以重写此方法
        return df

    def get_params(self) -> Dict:
        """
        获取策略参数

        Returns:
            参数字典
        """
        return self.params.copy()

    def set_param(self, key: str, value) -> None:
        """
        设置单个参数

        Args:
            key: 参数名
            value: 参数值
        """
        self.params[key] = value

    def reset(self) -> None:
        """
        重置策略状态
        用于多次回测或重新初始化
        """
        self.indicators = {}

    def __repr__(self) -> str:
        return f"{self.name}(params={self.params})"


class StrategyContext:
    """
    策略上下文类
    为策略提供额外的上下文信息
    """

    def __init__(self):
        self.current_date: Optional[date] = None
        self.portfolio_value: float = 0.0
        self.cash: float = 0.0
        self.positions: Dict[str, int] = {}  # {股票代码: 持仓数量}
        self.total_trades: int = 0
        self.win_trades: int = 0
        self.loss_trades: int = 0

    def update(self, date: date, portfolio) -> None:
        """
        更新上下文信息

        Args:
            date: 当前日期
            portfolio: 投资组合对象
        """
        self.current_date = date
        self.portfolio_value = portfolio.total_value
        self.cash = portfolio.cash
        self.positions = portfolio.positions.copy()

    def get_position_count(self) -> int:
        """获取当前持仓数量"""
        return len([pos for pos in self.positions.values() if pos.shares > 0])

    def get_position_ratio(self, code: str) -> float:
        """
        获取某只股票的仓位比例

        Args:
            code: 股票代码

        Returns:
            仓位比例（0-1）
        """
        if code not in self.positions:
            return 0.0

        if self.portfolio_value == 0:
            return 0.0

        # 这里需要获取股票的市值，暂时简化处理
        # 实际应该从 portfolio 对象获取
        return 0.0

    def is_in_position(self, code: str) -> bool:
        """
        检查是否持有某只股票

        Args:
            code: 股票代码

        Returns:
            是否持有
        """
        return code in self.positions and self.positions.get(code, 0) > 0

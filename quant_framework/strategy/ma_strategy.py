"""
双均线策略示例
基于短期和长期移动平均线的金叉和死叉产生买卖信号
"""

from datetime import date
from typing import List
import pandas as pd
from .base_strategy import BaseStrategy, Signal
from .indicators import Indicators
from ..data.data_handler import DataHandler
from ..portfolio.portfolio import Portfolio

class SimpleMAStrategy(BaseStrategy):
    """
    简单均线策略
    如果上一时间点价格高出五天平均价1%, 则全仓买入
    # 如果上一时间点价格低于五天平均价, 则空仓卖出
    """

    def __init__(self, params: dict = None):
        """
        初始化均线策略

        Args:
            params: 策略参数
                - window: 均线周期（默认10）
                - max_position: 最大持仓数量（默认3）
        """
        default_params = {
            'window': 10,
            'max_position': 3,
            'min_bars': 10  # 至少需要window条数据来计算均线
        }
        if params:
            default_params.update(params)

        super().__init__(default_params)

        self.window = self.params['window']
        self.max_position = self.params['max_position']

    def on_bar(self, data_handler: DataHandler, current_date: date, portfolio: Portfolio) -> List[Signal]:
        """
        每日策略逻辑

        Args:
            data_handler: 数据处理器
            current_date: 当前日期
            portfolio: 投资组合

        Returns:
            信号列表
        """
        signals = []

        # 获取所有股票代码（使用策略的股票列表或所有股票）
        codes = self.get_target_codes(data_handler)

        # 当前持仓数量
        current_position_count = len([pos for pos in portfolio.positions.values()
                                      if pos.shares > 0])

        for code in codes:
            # 获取历史数据（避免未来函数）
            df = data_handler.get_data_before(code, current_date)

            # 数据不足，跳过
            if len(df) < self.window + 1:
                continue

            # 计算均线
            df = self.calculate_indicators(df)

            today = df.iloc[-1]

            # 创建信号对象
            signal = Signal()
            signal.date = current_date
            signal.code = code
            signal.price = today['close']

            # 判断是否已持仓
            cash = portfolio.get_cash()
            position = portfolio.get_position(code)
            current_shares = position.shares if position else 0

            # 买入逻辑
            if today['close'] > today['MA'] * 1.01 and cash > 0:
                signal.signal_type = Signal.BUY
                signal.weight = 1.0  # 全部买入
                signal.reason = f"价格高于均价 1%, 买入 {code}"
                signals.append(signal)

            # 卖出逻辑
            elif today['close'] < today['MA'] and current_shares > 0:
                signal.signal_type = Signal.SELL
                signal.weight = 1.0  # 全部卖出
                signal.reason = f"价格低于均价, 卖出 {code}"
                signals.append(signal)

        return signals

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标

        Args:
            df: 价格数据

        Returns:
            添加了均线的数据
        """
        df = df.copy()

        # 计算均线
        df['MA'] = Indicators.sma(df, period=self.window)

        return df

class DoubleMAStrategy(BaseStrategy):
    """
    简单双均线策略
    当短期均线上穿长期均线时买入（金叉）
    当短期均线下穿长期均线时卖出（死叉）
    """

    def __init__(self, params: dict = None):
        """
        初始化双均线策略

        Args:
            params: 策略参数
                - short_window: 短期均线周期（默认10）
                - long_window: 长期均线周期（默认30）
                - max_position: 最大持仓数量（默认3）
        """
        default_params = {
            'short_window': 10,
            'long_window': 30,
            'max_position': 3,
            'min_bars': 30  # 至少需要long_window条数据来计算长期均线
        }
        if params:
            default_params.update(params)

        super().__init__(default_params)

        self.short_window = self.params['short_window']
        self.long_window = self.params['long_window']
        self.max_position = self.params['max_position']

    def on_bar(self, data_handler: DataHandler, current_date: date, portfolio) -> List[Signal]:
        """
        每日策略逻辑

        Args:
            data_handler: 数据处理器
            current_date: 当前日期
            portfolio: 投资组合

        Returns:
            信号列表
        """
        signals = []

        # 获取所有股票代码（使用策略的股票列表或所有股票）
        codes = self.get_target_codes(data_handler)

        # 当前持仓数量
        current_position_count = len([pos for pos in portfolio.positions.values()
                                      if pos.shares > 0])

        for code in codes:
            # 获取历史数据（避免未来函数）
            df = data_handler.get_data_before(code, current_date)

            # 数据不足，跳过
            if len(df) < self.long_window + 1:
                continue

            # 计算均线
            df = self.calculate_indicators(df)

            # 获取最新两天数据
            if len(df) < 2:
                continue

            today = df.iloc[-1]
            yesterday = df.iloc[-2]

            # 创建信号对象
            signal = Signal()
            signal.date = current_date
            signal.code = code
            signal.price = today['close']

            # 判断是否已持仓
            position = portfolio.positions.get(code)
            current_shares = position.shares if position else 0

            # 金叉买入
            is_golden_cross = (yesterday['MA_short'] <= yesterday['MA_long'] and
                               today['MA_short'] > today['MA_long'])

            # 死叉卖出
            is_death_cross = (yesterday['MA_short'] >= yesterday['MA_long'] and
                              today['MA_short'] < today['MA_long'])

            # 买入逻辑
            if is_golden_cross and current_shares == 0:
                # 检查是否达到最大持仓数
                if current_position_count < self.max_position:
                    signal.signal_type = Signal.BUY
                    signal.weight = 1.0 / self.max_position  # 平均分配仓位
                    signal.reason = (f"MA金叉: MA{self.short_window} "
                                   f"上穿MA{self.long_window}")
                    signals.append(signal)

            # 卖出逻辑
            elif is_death_cross and current_shares > 0:
                signal.signal_type = Signal.SELL
                signal.weight = 1.0  # 全部卖出
                signal.reason = (f"MA死叉: MA{self.short_window} "
                               f"下穿MA{self.long_window}")
                signals.append(signal)

        return signals

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标

        Args:
            df: 价格数据

        Returns:
            添加了均线的数据
        """
        df = df.copy()

        # 计算短期和长期均线
        df['MA_short'] = Indicators.sma(df, period=self.short_window)
        df['MA_long'] = Indicators.sma(df, period=self.long_window)

        return df


class MultiMAStrategy(BaseStrategy):
    """
    多均线策略
    使用三条均线判断趋势
    """

    def __init__(self, params: dict = None):
        """
        初始化多均线策略

        Args:
            params: 策略参数
                - ma1: 第一条均线周期（默认5）
                - ma2: 第二条均线周期（默认10）
                - ma3: 第三条均线周期（默认20）
                - max_position: 最大持仓数量（默认3）
        """
        default_params = {
            'ma1': 5,
            'ma2': 10,
            'ma3': 20,
            'max_position': 3,
            'min_bars': 20  # 至少需要ma3条数据来计算第三条均线
        }
        if params:
            default_params.update(params)

        super().__init__(default_params)

        self.ma1 = self.params['ma1']
        self.ma2 = self.params['ma2']
        self.ma3 = self.params['ma3']
        self.max_position = self.params['max_position']

    def on_bar(self, data_handler, current_date: date, portfolio) -> List[Signal]:
        """
        每日策略逻辑

        多头排列：MA1 > MA2 > MA3，买入
        空头排列：MA1 < MA2 < MA3，卖出
        """
        signals = []

        # 获取所有股票代码（使用策略的股票列表或所有股票）
        codes = self.get_target_codes(data_handler)
        current_position_count = len([pos for pos in portfolio.positions.values()
                                      if pos.shares > 0])

        for code in codes:
            df = data_handler.get_data_before(code, current_date)

            if len(df) < self.ma3 + 1:
                continue

            df = self.calculate_indicators(df)

            if len(df) < 1:
                continue

            today = df.iloc[-1]
            signal = Signal()
            signal.date = current_date
            signal.code = code
            signal.price = today['close']

            position = portfolio.positions.get(code)
            current_shares = position.shares if position else 0

            # 多头排列：买入
            is_bullish = (today['MA1'] > today['MA2'] and
                          today['MA2'] > today['MA3'])

            # 空头排列：卖出
            is_bearish = (today['MA1'] < today['MA2'] and
                          today['MA2'] < today['MA3'])

            if is_bullish and current_shares == 0:
                if current_position_count < self.max_position:
                    signal.signal_type = Signal.BUY
                    signal.weight = 1.0 / self.max_position
                    signal.reason = f"多头排列: MA{self.ma1}>MA{self.ma2}>MA{self.ma3}"
                    signals.append(signal)

            elif is_bearish and current_shares > 0:
                signal.signal_type = Signal.SELL
                signal.weight = 1.0
                signal.reason = f"空头排列: MA{self.ma1}<MA{self.ma2}<MA{self.ma3}"
                signals.append(signal)

        return signals

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算三条均线"""
        df = df.copy()
        df['MA1'] = Indicators.sma(df, period=self.ma1)
        df['MA2'] = Indicators.sma(df, period=self.ma2)
        df['MA3'] = Indicators.sma(df, period=self.ma3)
        return df

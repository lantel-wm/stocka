"""
自定义策略示例
展示如何创建和使用自定义策略
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import date
from typing import List
import pandas as pd

from quant_framework import (
    DataHandler,
    BaseStrategy,
    Signal,
    BacktestEngine,
    StandardCost
)
from quant_framework.strategy.indicators import Indicators


class BreakoutStrategy(BaseStrategy):
    """
    突破策略示例
    当价格突破N日高点时买入
    """

    def __init__(self, params: dict = None):
        default_params = {
            'lookback_period': 20,    # 回看周期
            'breakout_ratio': 1.02,   # 突破比例
            'max_position': 3,        # 最大持仓数
            'min_bars': 20            # 至少需要lookback_period条数据来计算突破价格
        }
        if params:
            default_params.update(params)

        super().__init__(default_params)

        self.lookback_period = self.params['lookback_period']
        self.breakout_ratio = self.params['breakout_ratio']
        self.max_position = self.params['max_position']

    def on_bar(self, data_handler, current_date: date, portfolio) -> List[Signal]:
        """每日策略逻辑"""
        signals = []

        # 获取所有股票代码（使用策略的股票列表或所有股票）
        codes = self.get_target_codes(data_handler)
        current_position_count = len([pos for pos in portfolio.positions.values()
                                      if pos.shares > 0])

        for code in codes:
            # 获取历史数据
            df = data_handler.get_data_before(code, current_date)

            if len(df) < self.lookback_period + 1:
                continue

            # 只使用历史数据
            df = df.tail(self.lookback_period)

            # 获取最新数据
            current_data = data_handler.get_daily_data(current_date)
            if code not in current_data.index:
                continue

            current_price = current_data.loc[code, 'close']

            # 计算突破价格
            highest_price = df['high'].max()
            breakout_price = highest_price * self.breakout_ratio

            # 创建信号
            signal = Signal()
            signal.date = current_date
            signal.code = code
            signal.price = current_price

            position = portfolio.positions.get(code)
            current_shares = position.shares if position else 0

            # 突破买入
            if current_price > breakout_price and current_shares == 0:
                if current_position_count < self.max_position:
                    signal.signal_type = Signal.BUY
                    signal.weight = 1.0 / self.max_position
                    signal.reason = (f"突破{self.lookback_period}日高点 "
                                   f"({current_price:.2f} > {breakout_price:.2f})")
                    signals.append(signal)

            # 简单的止损退出
            elif current_shares > 0:
                # 如果跌破5日均线，卖出
                if len(df) >= 5:
                    ma5 = df['close'].tail(5).mean()
                    if current_price < ma5:
                        signal.signal_type = Signal.SELL
                        signal.weight = 1.0
                        signal.reason = f"跌破5日均线 ({current_price:.2f} < {ma5:.2f})"
                        signals.append(signal)

        return signals


class MomentumStrategy(BaseStrategy):
    """
    动量策略示例
    基于RSI指标
    """

    def __init__(self, params: dict = None):
        default_params = {
            'rsi_period': 14,
            'oversold_threshold': 30,   # 超卖阈值
            'overbought_threshold': 70,  # 超买阈值
            'max_position': 3,
            'min_bars': 14              # 至少需要rsi_period条数据来计算RSI
        }
        if params:
            default_params.update(params)

        super().__init__(default_params)

        self.rsi_period = self.params['rsi_period']
        self.oversold_threshold = self.params['oversold_threshold']
        self.overbought_threshold = self.params['overbought_threshold']
        self.max_position = self.params['max_position']

    def on_bar(self, data_handler, current_date: date, portfolio) -> List[Signal]:
        """每日策略逻辑"""
        signals = []

        # 获取所有股票代码（使用策略的股票列表或所有股票）
        codes = self.get_target_codes(data_handler)
        current_position_count = len([pos for pos in portfolio.positions.values()
                                      if pos.shares > 0])

        for code in codes:
            df = data_handler.get_data_before(code, current_date)

            if len(df) < self.rsi_period + 1:
                continue

            # 计算RSI
            rsi = Indicators.rsi(df, period=self.rsi_period)

            if rsi.isna().all():
                continue

            current_rsi = rsi.iloc[-1]

            # 获取当前价格
            current_data = data_handler.get_daily_data(current_date)
            if code not in current_data.index:
                continue

            current_price = current_data.loc[code, 'close']

            signal = Signal()
            signal.date = current_date
            signal.code = code
            signal.price = current_price

            position = portfolio.positions.get(code)
            current_shares = position.shares if position else 0

            # RSI超卖买入
            if (current_rsi < self.oversold_threshold and
                current_shares == 0):
                if current_position_count < self.max_position:
                    signal.signal_type = Signal.BUY
                    signal.weight = 1.0 / self.max_position
                    signal.reason = f"RSI超卖 ({current_rsi:.2f} < {self.oversold_threshold})"
                    signals.append(signal)

            # RSI超买卖出
            elif (current_rsi > self.overbought_threshold and
                  current_shares > 0):
                signal.signal_type = Signal.SELL
                signal.weight = 1.0
                signal.reason = f"RSI超买 ({current_rsi:.2f} > {self.overbought_threshold})"
                signals.append(signal)

        return signals


def main():
    """主函数"""
    print("=" * 70)
    print("自定义策略示例")
    print("=" * 70)
    print()

    # 加载数据
    print("加载数据...")
    data_handler = DataHandler(
        data_path="data/stock/kline/day",
        min_data_points=100
    )

    try:
        data_handler.load_data(
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
    except Exception as e:
        print(f"加载数据失败：{e}")
        return

    # 示例1：突破策略
    print("\n示例1：突破策略回测")
    print("-" * 70)

    strategy1 = BreakoutStrategy({
        'lookback_period': 20,
        'breakout_ratio': 1.02,
        'max_position': 3
    })

    engine1 = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy1,
        initial_capital=1000000,
        transaction_cost=StandardCost()
    )

    results1 = engine1.run(
        start_date="2020-01-01",
        end_date="2023-12-31",
        verbose=False
    )

    print(f"最终权益: {results1['final_value']:,.2f} 元")
    print(f"总收益率: {results1['total_return']*100:.2f}%")
    print(f"交易次数: {len(results1['trades'])}")

    # 示例2：动量策略
    print("\n示例2：动量策略回测")
    print("-" * 70)

    strategy2 = MomentumStrategy({
        'rsi_period': 14,
        'oversold_threshold': 30,
        'overbought_threshold': 70,
        'max_position': 3
    })

    engine2 = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy2,
        initial_capital=1000000,
        transaction_cost=StandardCost()
    )

    results2 = engine2.run(
        start_date="2020-01-01",
        end_date="2023-12-31",
        verbose=False
    )

    print(f"最终权益: {results2['final_value']:,.2f} 元")
    print(f"总收益率: {results2['total_return']*100:.2f}%")
    print(f"交易次数: {len(results2['trades'])}")

    print("\n" + "=" * 70)
    print("自定义策略示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

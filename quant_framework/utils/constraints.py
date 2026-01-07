"""
交易限制处理模块
处理A股的各种交易限制（涨跌停、停牌、T+1等）
"""

from datetime import date
from typing import Optional
import pandas as pd


class TradingConstraints:
    """
    交易限制处理类
    检查是否符合A股交易规则
    """

    @staticmethod
    def is_limit_up(price: float, pre_close: float, limit_rate: float = 0.10) -> bool:
        """
        判断是否涨停

        Args:
            price: 当前价格
            pre_close: 前收盘价
            limit_rate: 涨停限制（主板默认10%，创业板20%）

        Returns:
            是否涨停
        """
        if pre_close == 0:
            return False
        return price >= pre_close * (1 + limit_rate) * 0.9995

    @staticmethod
    def is_limit_down(price: float, pre_close: float, limit_rate: float = 0.10) -> bool:
        """
        判断是否跌停

        Args:
            price: 当前价格
            pre_close: 前收盘价
            limit_rate: 跌停限制（主板默认10%，创业板20%）

        Returns:
            是否跌停
        """
        if pre_close == 0:
            return False
        return price <= pre_close * (1 - limit_rate) * 1.0005

    @staticmethod
    def is_suspended(df: pd.DataFrame, trade_date: date) -> bool:
        """
        判断是否停牌

        Args:
            df: 股票数据
            trade_date: 交易日期

        Returns:
            是否停牌
        """
        # 将date转换为datetime类型进行比较
        if isinstance(trade_date, date):
            trade_date = pd.Timestamp(trade_date)

        # 检查是否有当日数据
        if df.index.name == 'date':
            has_data = trade_date in df.index
        elif 'date' in df.columns:
            has_data = trade_date in pd.to_datetime(df['date']).values
        else:
            # 如果没有日期信息，假设没有停牌
            has_data = True

        return not has_data

    @staticmethod
    def can_trade(code: str,
                  trade_date: date,
                  data_handler,
                  signal_type: str) -> bool:
        """
        判断是否可以交易

        考虑涨跌停和停牌情况

        Args:
            code: 股票代码
            trade_date: 交易日期
            data_handler: 数据处理器
            signal_type: 信号类型（'buy' or 'sell'）

        Returns:
            是否可以交易
        """
        # 获取股票数据
        df = data_handler.get_stock_data(code)

        if df.empty:
            return False

        # 检查是否停牌
        if TradingConstraints.is_suspended(df, trade_date):
            return False

        # 获取当日数据
        try:
            if df.index.name == 'date':
                daily_data = df.loc[pd.Timestamp(trade_date)]
            else:
                daily_data = df[df['date'] == pd.Timestamp(trade_date)]
                if daily_data.empty:
                    return False
                daily_data = daily_data.iloc[0]
        except (KeyError, IndexError):
            return False

        # 获取前收盘价
        pre_close = daily_data.get('pre_close', 0)
        if pre_close == 0:
            # 如果没有前收盘价，使用前一天的收盘价
            try:
                pre_close = df['close'].shift(1).loc[pd.Timestamp(trade_date)]
            except (KeyError, IndexError):
                return False

        current_price = daily_data.get('close', daily_data.get('收盘', 0))

        if current_price == 0:
            return False

        # 买入时检查涨停
        if signal_type == 'buy':
            if TradingConstraints.is_limit_up(current_price, pre_close):
                return False

        # 卖出时检查跌停
        elif signal_type == 'sell':
            if TradingConstraints.is_limit_down(current_price, pre_close):
                return False

        return True

    @staticmethod
    def is_new_stock(list_date: Optional[str],
                    trade_date: date,
                    days: int = 5) -> bool:
        """
        判断是否为新上市股票（前N个交易日无涨跌停限制）

        Args:
            list_date: 上市日期
            trade_date: 当前交易日期
            days: 天数（默认5天）

        Returns:
            是否为新股
        """
        if not list_date:
            return False

        try:
            listing = pd.to_datetime(list_date).date()
            delta = (trade_date - listing).days
            return delta <= days
        except:
            return False

    @staticmethod
    def is_st_stock(code: str) -> bool:
        """
        判断是否为ST股票（特别处理股票）

        Args:
            code: 股票代码

        Returns:
            是否为ST股票
        """
        # ST股票的命名规则：股票名称中包含ST、*ST、S*ST等
        # 这里需要股票名称数据，暂时返回False
        # 实际实现时应该从股票列表中获取名称并判断
        return False

    @staticmethod
    def get_limit_rate(code: str) -> float:
        """
        获取涨跌停限制比例

        Args:
            code: 股票代码

        Returns:
            涨跌停限制比例
        """
        # 主板：10%
        # 创业板/科创板：20%
        # ST股票：5%

        if TradingConstraints.is_st_stock(code):
            return 0.05

        # 创业板股票代码以300开头
        # 科创板股票代码以688开头
        if code.startswith('300') or code.startswith('688'):
            return 0.20

        # 主板默认10%
        return 0.10


class LiquidityChecker:
    """流动性检查"""

    @staticmethod
    def check_liquidity(df: pd.DataFrame,
                       trade_date: date,
                       trade_amount: float,
                       max_ratio: float = 0.1) -> bool:
        """
        检查流动性

        确保交易金额不超过当日成交额的一定比例

        Args:
            df: 股票数据
            trade_date: 交易日期
            trade_amount: 交易金额
            max_ratio: 最大成交额比例（默认10%）

        Returns:
            流动性是否充足
        """
        try:
            if df.index.name == 'date':
                daily_data = df.loc[pd.Timestamp(trade_date)]
            else:
                daily_data = df[df['date'] == pd.Timestamp(trade_date)]
                if daily_data.empty:
                    return False
                daily_data = daily_data.iloc[0]

            daily_amount = daily_data.get('amount', daily_data.get('成交额', 0))

            if daily_amount == 0:
                return False

            return trade_amount <= daily_amount * max_ratio

        except (KeyError, IndexError):
            return False

    @staticmethod
    def calculate_max_shares(df: pd.DataFrame,
                            trade_date: date,
                            cash: float,
                            price: float,
                            max_ratio: float = 0.1) -> int:
        """
        计算最大可买入股数

        考虑：
        1. 资金限制
        2. 流动性限制
        3. A股交易单位（100股为一手）

        Args:
            df: 股票数据
            trade_date: 交易日期
            cash: 可用资金
            price: 价格
            max_ratio: 最大成交额比例

        Returns:
            最大可买入股数
        """
        try:
            if df.index.name == 'date':
                daily_data = df.loc[pd.Timestamp(trade_date)]
            else:
                daily_data = df[df['date'] == pd.Timestamp(trade_date)]
                if daily_data.empty:
                    return 0
                daily_data = daily_data.iloc[0]

            daily_amount = daily_data.get('amount', daily_data.get('成交额', 0))

            # 基于流动性的最大金额
            if daily_amount > 0:
                max_amount_by_liquidity = daily_amount * max_ratio
            else:
                max_amount_by_liquidity = float('inf')

            # 基于资金的最大金额
            max_amount_by_cash = cash

            # 取较小值
            max_amount = min(max_amount_by_liquidity, max_amount_by_cash)

            # 计算股数（向下取整到100的倍数）
            max_shares = int(max_amount / price / 100) * 100

            return max_shares

        except (KeyError, IndexError):
            return 0


class TradingCalendar:
    """交易日历"""

    @staticmethod
    def is_trading_day(trade_date: date, trading_days: list) -> bool:
        """
        判断是否为交易日

        Args:
            trade_date: 日期
            trading_days: 交易日列表

        Returns:
            是否为交易日
        """
        return trade_date in trading_days

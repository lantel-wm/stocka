"""
技术指标库
提供常用的技术分析指标计算函数
"""

from typing import Union, Tuple
import pandas as pd
import numpy as np


class Indicators:
    """
    技术指标计算类
    提供静态方法计算各种技术指标
    """

    @staticmethod
    def sma(df: pd.DataFrame, column: str = 'close', period: int = 20) -> pd.Series:
        """
        简单移动平均线 (Simple Moving Average)

        Args:
            df: 数据DataFrame
            column: 计算列名
            period: 周期

        Returns:
            SMA序列
        """
        return df[column].rolling(window=period).mean()

    @staticmethod
    def ema(df: pd.DataFrame, column: str = 'close', period: int = 20) -> pd.Series:
        """
        指数移动平均线 (Exponential Moving Average)

        Args:
            df: 数据DataFrame
            column: 计算列名
            period: 周期

        Returns:
            EMA序列
        """
        return df[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    def macd(df: pd.DataFrame,
             column: str = 'close',
             fast_period: int = 12,
             slow_period: int = 26,
             signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD指标 (Moving Average Convergence Divergence)

        Args:
            df: 数据DataFrame
            column: 计算列名
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期

        Returns:
            (MACD线, 信号线, 柱状图)
        """
        ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def rsi(df: pd.DataFrame,
            column: str = 'close',
            period: int = 14) -> pd.Series:
        """
        相对强弱指标 (Relative Strength Index)

        Args:
            df: 数据DataFrame
            column: 计算列名
            period: 周期

        Returns:
            RSI序列
        """
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def bollinger_bands(df: pd.DataFrame,
                        column: str = 'close',
                        period: int = 20,
                        std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        布林带 (Bollinger Bands)

        Args:
            df: 数据DataFrame
            column: 计算列名
            period: 周期
            std_dev: 标准差倍数

        Returns:
            (上轨, 中轨, 下轨)
        """
        middle_band = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()

        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        return upper_band, middle_band, lower_band

    @staticmethod
    def stochastic(df: pd.DataFrame,
                   period: int = 14,
                   smooth_k: int = 3,
                   smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        随机指标 (Stochastic Oscillator)

        Args:
            df: 数据DataFrame
            period: 周期
            smooth_k: K值平滑
            smooth_d: D值平滑

        Returns:
            (K值, D值)
        """
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()

        k_percent = 100 * (df['close'] - low_min) / (high_max - low_min)
        k_smooth = k_percent.rolling(window=smooth_k).mean()
        d_smooth = k_smooth.rolling(window=smooth_d).mean()

        return k_smooth, d_smooth

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        平均真实波幅 (Average True Range)

        Args:
            df: 数据DataFrame
            period: 周期

        Returns:
            ATR序列
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """
        能量潮指标 (On Balance Volume)

        Args:
            df: 数据DataFrame

        Returns:
            OBV序列
        """
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        顺势指标 (Commodity Channel Index)

        Args:
            df: 数据DataFrame
            period: 周期

        Returns:
            CCI序列
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )

        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        威廉指标 (Williams %R)

        Args:
            df: 数据DataFrame
            period: 周期

        Returns:
            Williams %R序列
        """
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()

        williams_r = -100 * (high_max - df['close']) / (high_max - low_min)
        return williams_r

    @staticmethod
    def volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        成交量移动平均

        Args:
            df: 数据DataFrame
            period: 周期

        Returns:
            成交量MA序列
        """
        return df['volume'].rolling(window=period).mean()

    @staticmethod
    def price_change_rate(df: pd.DataFrame,
                          column: str = 'close',
                          period: int = 1) -> pd.Series:
        """
        价格变化率

        Args:
            df: 数据DataFrame
            column: 计算列名
            period: 周期

        Returns:
            变化率序列
        """
        return df[column].pct_change(period)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    为数据添加所有常用技术指标

    Args:
        df: 原始OHLCV数据

    Returns:
        添加了所有指标的数据
    """
    df = df.copy()

    # 移动平均线
    df['MA5'] = Indicators.sma(df, period=5)
    df['MA10'] = Indicators.sma(df, period=10)
    df['MA20'] = Indicators.sma(df, period=20)
    df['MA30'] = Indicators.sma(df, period=30)
    df['MA60'] = Indicators.sma(df, period=60)

    # EMA
    df['EMA12'] = Indicators.ema(df, period=12)
    df['EMA26'] = Indicators.ema(df, period=26)

    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = Indicators.macd(df)

    # RSI
    df['RSI'] = Indicators.rsi(df)

    # 布林带
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = Indicators.bollinger_bands(df)

    # ATR
    df['ATR'] = Indicators.atr(df)

    # 成交量MA
    df['Volume_MA5'] = Indicators.volume_ma(df, period=5)
    df['Volume_MA20'] = Indicators.volume_ma(df, period=20)

    return df

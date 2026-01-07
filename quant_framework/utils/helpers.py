"""
辅助工具函数模块
提供各种通用的辅助函数
"""

from datetime import date, datetime, timedelta
from typing import List, Optional
import pandas as pd
import numpy as np


def date_to_str(dt: date) -> str:
    """
    将日期转换为字符串

    Args:
        dt: 日期对象

    Returns:
        日期字符串 (YYYY-MM-DD)
    """
    return dt.strftime('%Y-%m-%d')


def str_to_date(date_str: str) -> date:
    """
    将字符串转换为日期

    Args:
        date_str: 日期字符串 (YYYY-MM-DD)

    Returns:
        日期对象
    """
    return datetime.strptime(date_str, '%Y-%m-%d').date()


def calculate_date_range(start_date: str,
                         end_date: str) -> List[date]:
    """
    生成日期范围

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        日期列表
    """
    start = str_to_date(start_date)
    end = str_to_date(end_date)

    delta = end - start
    dates = [start + timedelta(days=i) for i in range(delta.days + 1)]

    return dates


def filter_trading_days(dates: List[date],
                       trading_days: List[date]) -> List[date]:
    """
    过滤出交易日

    Args:
        dates: 所有日期
        trading_days: 交易日列表

    Returns:
        交易日列表
    """
    trading_set = set(trading_days)
    return [d for d in dates if d in trading_set]


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    计算收益率

    Args:
        prices: 价格序列

    Returns:
        收益率序列
    """
    return prices.pct_change()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    计算累计收益率

    Args:
        returns: 收益率序列

    Returns:
        累计收益率序列
    """
    return (1 + returns).cumprod()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    计算对数收益率

    Args:
        prices: 价格序列

    Returns:
        对数收益率序列
    """
    return np.log(prices / prices.shift(1))


def annualize_return(daily_return: float,
                    trading_days: int = 252) -> float:
    """
    年化收益率

    Args:
        daily_return: 日收益率
        trading_days: 年化交易日数

    Returns:
        年化收益率
    """
    return (1 + daily_return) ** trading_days - 1


def annualize_volatility(daily_volatility: float,
                        trading_days: int = 252) -> float:
    """
    年化波动率

    Args:
        daily_volatility: 日波动率
        trading_days: 年化交易日数

    Returns:
        年化波动率
    """
    return daily_volatility * np.sqrt(trading_days)


def calculate_max_drawdown(equity_curve: pd.Series) -> tuple:
    """
    计算最大回撤

    Args:
        equity_curve: 权益曲线

    Returns:
        (最大回撤, 回撤开始日期, 回撤结束日期)
    """
    cumulative = equity_curve.cummax()
    drawdown = (equity_curve - cumulative) / cumulative

    max_dd = drawdown.min()

    # 找到最大回撤的日期
    max_dd_date = drawdown.idxmin()

    # 找到回撤开始的高点日期
    peak_date = equity_curve[:max_dd_date].idxmax()

    return max_dd, peak_date, max_dd_date


def calculate_sharpe_ratio(returns: pd.Series,
                           risk_free_rate: float = 0.03,
                           trading_days: int = 252) -> float:
    """
    计算夏普比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        trading_days: 年化交易日数

    Returns:
        夏普比率
    """
    daily_rf = risk_free_rate / trading_days
    excess_returns = returns - daily_rf

    if excess_returns.std() == 0:
        return 0.0

    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days)
    return sharpe


def calculate_sortino_ratio(returns: pd.Series,
                           risk_free_rate: float = 0.03,
                           trading_days: int = 252) -> float:
    """
    计算索提诺比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        trading_days: 年化交易日数

    Returns:
        索提诺比率
    """
    daily_rf = risk_free_rate / trading_days
    excess_returns = returns - daily_rf

    # 只考虑负收益的标准差（下行风险）
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(trading_days)
    return sortino


def calculate_win_rate(trades: list) -> float:
    """
    计算胜率

    Args:
        trades: 交易记录列表

    Returns:
        胜率（0-1）
    """
    if not trades:
        return 0.0

    winning_trades = [t for t in trades if t.get('profit', 0) > 0]
    return len(winning_trades) / len(trades)


def calculate_profit_loss_ratio(trades: list) -> float:
    """
    计算盈亏比

    Args:
        trades: 交易记录列表

    Returns:
        盈亏比
    """
    winning_trades = [t.get('profit', 0) for t in trades if t.get('profit', 0) > 0]
    losing_trades = [abs(t.get('profit', 0)) for t in trades if t.get('profit', 0) < 0]

    if not winning_trades or not losing_trades:
        return 0.0

    avg_win = np.mean(winning_trades)
    avg_loss = np.mean(losing_trades)

    if avg_loss == 0:
        return float('inf')

    return avg_win / avg_loss


def format_number(value: float,
                 decimals: int = 2,
                 prefix: str = "",
                 suffix: str = "") -> str:
    """
    格式化数字

    Args:
        value: 数值
        decimals: 小数位数
        prefix: 前缀
        suffix: 后缀

    Returns:
        格式化后的字符串
    """
    return f"{prefix}{value:.{decimals}f}{suffix}"


def format_percent(value: float, decimals: int = 2) -> str:
    """
    格式化百分比

    Args:
        value: 数值（0-1）
        decimals: 小数位数

    Returns:
        格式化后的百分比字符串
    """
    return f"{value * 100:.{decimals}f}%"


def load_config(config_path: str) -> dict:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: dict, config_path: str) -> None:
    """
    保存配置文件

    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    import yaml

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，不存在则创建

    Args:
        directory: 目录路径
    """
    import os

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_current_time_str() -> str:
    """
    获取当前时间字符串

    Returns:
        时间字符串 (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def chunks(lst: list, n: int) -> list:
    """
    将列表分成大小为n的块

    Args:
        lst: 列表
        n: 块大小

    Returns:
        嵌套列表
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]

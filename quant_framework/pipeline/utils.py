"""
Pipeline工具函数
"""

from datetime import datetime, date
import pandas as pd


def str_to_date(date_str: str) -> date:
    """将字符串转换为日期对象"""
    return datetime.strptime(date_str, '%Y-%m-%d').date()


def calculate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算标签（未来收益率）
    LABEL0 = close(-2) / close(-1) - 1

    Args:
        df: 包含close列的DataFrame，MultiIndex为(code, date)

    Returns:
        添加了LABEL0列的DataFrame
    """
    df = df.copy()

    # 按 code 分组计算未来收益率
    # shift(-1): 取下一个交易日的收盘价
    # shift(-2): 取下下个交易日的收盘价
    close_shift_1 = df.groupby(level=0)['close'].shift(-1)
    close_shift_2 = df.groupby(level=0)['close'].shift(-2)

    # 计算收益率: 第2天的收盘价 / 第1天的收盘价 - 1
    df['LABEL0'] = close_shift_2 / close_shift_1 - 1

    return df

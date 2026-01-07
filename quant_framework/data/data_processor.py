"""
数据预处理与清洗模块
负责数据清洗、前收盘价计算等预处理工作
"""

from typing import Optional
import pandas as pd
import numpy as np


class DataProcessor:
    """
    数据预处理器
    提供数据清洗和预处理的静态方法
    """

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗

        Args:
            df: 原始数据DataFrame

        Returns:
            清洗后的DataFrame
        """
        # 创建副本避免修改原数据
        df = df.copy()

        # 1. 去除关键列的缺失值
        required_columns = ['date', 'open', 'high', 'low', 'close']
        if 'code' in df.columns:
            required_columns.append('code')

        df = df.dropna(subset=required_columns)

        # 2. 处理异常值（价格为0或负数）
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df = df[df[col] > 0]

        # 3. 确保最高价 >= 最低价
        if 'high' in df.columns and 'low' in df.columns:
            df = df[df['high'] >= df['low']]

        # 4. 确保收盘价在最高价和最低价之间
        if all(col in df.columns for col in ['close', 'high', 'low']):
            df = df[(df['close'] <= df['high']) & (df['close'] >= df['low'])]

        # 5. 确保开盘价在最高价和最低价之间
        if all(col in df.columns for col in ['open', 'high', 'low']):
            df = df[(df['open'] <= df['high']) & (df['open'] >= df['low'])]

        # 6. 去除成交量为0或负数的记录
        if 'volume' in df.columns:
            df = df[df['volume'] > 0]

        # 7. 去除成交额为0或负数的记录
        if 'amount' in df.columns:
            df = df[df['amount'] > 0]

        # 8. 重置索引
        df = df.reset_index(drop=True)

        return df

    @staticmethod
    def calculate_pre_close(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算前收盘价（用于计算涨跌幅）

        Args:
            df: 包含收盘价的数据

        Returns:
            添加了 pre_close 列的DataFrame
        """
        df = df.copy()

        # 确保按日期排序
        if 'date' in df.columns:
            df = df.sort_values('date')
        elif df.index.name == 'date':
            df = df.sort_index()

        # 计算前收盘价（前一天收盘价）
        df['pre_close'] = df['close'].shift(1)

        # 第一条记录的前收盘价设为收盘价（避免NaN）
        df.loc[df.index[0], 'pre_close'] = df.loc[df.index[0], 'close']

        return df

    @staticmethod
    def calculate_pct_change(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算涨跌幅

        Args:
            df: 包含收盘价和前收盘价的数据

        Returns:
            添加了 pct_change 列的DataFrame
        """
        df = df.copy()

        # 如果没有前收盘价，先计算
        if 'pre_close' not in df.columns:
            df = DataProcessor.calculate_pre_close(df)

        # 计算涨跌幅
        df['pct_change'] = (df['close'] - df['pre_close']) / df['pre_close']

        return df

    @staticmethod
    def validate_data(df: pd.DataFrame) -> dict:
        """
        验证数据质量

        Args:
            df: 要验证的数据

        Returns:
            验证结果字典
        """
        result = {
            "valid": True,
            "issues": [],
            "warnings": []
        }

        # 检查必需列
        required_columns = ['date', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            result["valid"] = False
            result["issues"].append(f"缺少必需列: {missing_columns}")

        # 检查缺失值
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            null_columns = null_counts[null_counts > 0].to_dict()
            result["warnings"].append(f"存在缺失值: {null_columns}")

        # 检查异常值
        if 'high' in df.columns and 'low' in df.columns:
            if (df['high'] < df['low']).any():
                result["valid"] = False
                result["issues"].append("存在最高价小于最低价的异常记录")

        if 'close' in df.columns and 'high' in df.columns and 'low' in df.columns:
            if ((df['close'] > df['high']) | (df['close'] < df['low'])).any():
                result["valid"] = False
                result["issues"].append("存在收盘价超出最高最低价范围的异常记录")

        # 检查数据量
        if len(df) < 2:
            result["warnings"].append("数据量较少，可能影响分析结果")

        return result

    @staticmethod
    def resample_data(df: pd.DataFrame,
                      freq: str = 'D') -> pd.DataFrame:
        """
        重采样数据（如将日线数据转换为周线或月线）

        Args:
            df: 原始日线数据
            freq: 重采样频率 ('D'=日, 'W'=周, 'M'=月)

        Returns:
            重采样后的数据
        """
        df = df.copy()

        # 确保日期是索引或列
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        # 重采样
        if freq == 'D':
            # 日线不需要重采样
            return df
        elif freq == 'W':
            # 周线：使用周五的数据
            resampled = df.resample('W-FRI').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'amount': 'sum'
            })
        elif freq == 'M':
            # 月线：使用月末的数据
            resampled = df.resample('M').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'amount': 'sum'
            })
        else:
            raise ValueError(f"不支持的重采样频率: {freq}")

        # 去除全为NaN的行
        resampled = resampled.dropna(how='all')

        return resampled

    @staticmethod
    def filter_by_date(df: pd.DataFrame,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        按日期范围过滤数据

        Args:
            df: 原始数据
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            过滤后的数据
        """
        df = df.copy()

        # 确保日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 过滤
        if 'date' in df.columns:
            if start_date:
                df = df[df['date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['date'] <= pd.to_datetime(end_date)]
        else:
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]

        return df

    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        添加基础特征

        Args:
            df: 原始数据

        Returns:
            添加了基础特征的数据
        """
        df = df.copy()

        # 确保有前收盘价和涨跌幅
        if 'pre_close' not in df.columns:
            df = DataProcessor.calculate_pre_close(df)

        if 'pct_change' not in df.columns:
            df = DataProcessor.calculate_pct_change(df)

        # 计算振幅
        if all(col in df.columns for col in ['high', 'low', 'pre_close']):
            df['amplitude'] = (df['high'] - df['low']) / df['pre_close']

        # 计算涨跌额
        if all(col in df.columns for col in ['close', 'pre_close']):
            df['change_amount'] = df['close'] - df['pre_close']

        # 计算换手率（如果有流通股本数据）
        # TODO: 需要流通股本数据才能计算

        return df

    @staticmethod
    def remove_outliers(df: pd.DataFrame,
                        column: str = 'close',
                        n_std: float = 3.0) -> pd.DataFrame:
        """
        移除异常值（使用标准差方法）

        Args:
            df: 原始数据
            column: 要检查的列名
            n_std: 标准差倍数

        Returns:
            移除异常值后的数据
        """
        df = df.copy()

        if column not in df.columns:
            return df

        # 计算均值和标准差
        mean = df[column].mean()
        std = df[column].std()

        # 定义正常范围
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std

        # 过滤异常值
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        return df

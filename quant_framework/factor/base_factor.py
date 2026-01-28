"""
因子基类
定义因子的基本接口和通用方法
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np


class BaseFactor(ABC):
    """
    因子基类

    所有因子都应该继承此类并实现 calculate() 方法
    每个因子类可以返回多个因子列，提高计算效率
    """

    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        初始化因子

        Args:
            name: 因子名称（或因子组名称）
            params: 因子参数
        """
        self.name = name
        self.params = params or {}

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子值，向输入的DataFrame中添加因子列

        Args:
            data: 包含OHLCV数据的DataFrame，至少包含open, high, low, close, volume列

        Returns:
            包含原始数据和新因子列的DataFrame
        """
        pass

    # ==================== 辅助方法 ====================

    @staticmethod
    def _calculate_slope(series: pd.Series, window: int) -> pd.Series:
        """计算线性回归斜率"""
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window - 1, len(series)):
            y = series.iloc[i - window + 1:i + 1].values
            x = np.arange(window)
            if len(y) == window and np.isfinite(y).all():
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope
        return slopes

    @staticmethod
    def _calculate_rsquare(series: pd.Series, window: int) -> pd.Series:
        """计算线性回归R平方"""
        rsquares = pd.Series(index=series.index, dtype=float)
        for i in range(window - 1, len(series)):
            y = series.iloc[i - window + 1:i + 1].values
            x = np.arange(window)
            if len(y) == window and np.isfinite(y).all():
                coef = np.polyfit(x, y, 1)
                y_pred = np.poly1d(coef)(x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                rsqr = 1 - (ss_res / (ss_tot + 1e-12))
                rsquares.iloc[i] = rsqr
        return rsquares

    @staticmethod
    def _calculate_residual(series: pd.Series, window: int) -> pd.Series:
        """计算线性回归残差"""
        residuals = pd.Series(index=series.index, dtype=float)
        for i in range(window - 1, len(series)):
            y = series.iloc[i - window + 1:i + 1].values
            x = np.arange(window)
            if len(y) == window and np.isfinite(y).all():
                coef = np.polyfit(x, y, 1)
                y_pred = np.poly1d(coef)(x)
                residual = y[-1] - y_pred[-1]
                residuals.iloc[i] = residual
        return residuals

    @staticmethod
    def _calculate_idxmax(series: pd.Series, window: int) -> pd.Series:
        """计算滚动窗口内最大值的位置索引"""
        idxmax = pd.Series(index=series.index, dtype=float)
        for i in range(window - 1, len(series)):
            window_data = series.iloc[i - window + 1:i + 1]
            if len(window_data) == window and np.isfinite(window_data).all():
                max_idx = window_data.argmax()
                idxmax.iloc[i] = window - 1 - max_idx
        return idxmax

    @staticmethod
    def _calculate_idxmin(series: pd.Series, window: int) -> pd.Series:
        """计算滚动窗口内最小值的位置索引"""
        idxmin = pd.Series(index=series.index, dtype=float)
        for i in range(window - 1, len(series)):
            window_data = series.iloc[i - window + 1:i + 1]
            if len(window_data) == window and np.isfinite(window_data).all():
                min_idx = window_data.argmin()
                idxmin.iloc[i] = window - 1 - min_idx
        return idxmin

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证数据是否满足计算要求

        Args:
            data: 输入数据

        Returns:
            是否满足要求
        """
        required_columns = ['close']
        return all(col in data.columns for col in required_columns)

    def handle_missing_values(self, factor_values: pd.Series,
                              method: str = 'ffill') -> pd.Series:
        """
        处理缺失值

        Args:
            factor_values: 因子值
            method: 处理方法 ('ffill', 'bfill', 'drop', 'mean')

        Returns:
            处理后的因子值
        """
        if method == 'ffill':
            return factor_values.ffill()
        elif method == 'bfill':
            return factor_values.bfill()
        elif method == 'drop':
            return factor_values.dropna()
        elif method == 'mean':
            mean_value = factor_values.mean()
            return factor_values.fillna(mean_value)
        else:
            return factor_values

    def standardize(self, factor_values: pd.Series,
                    method: str = 'zscore') -> pd.Series:
        """
        标准化因子值

        Args:
            factor_values: 原始因子值
            method: 标准化方法 ('zscore', 'minmax', 'rank')

        Returns:
            标准化后的因子值
        """
        if method == 'zscore':
            # Z-score标准化
            mean = factor_values.mean()
            std = factor_values.std()
            return (factor_values - mean) / std

        elif method == 'minmax':
            # Min-Max标准化到[0, 1]
            min_val = factor_values.min()
            max_val = factor_values.max()
            return (factor_values - min_val) / (max_val - min_val)

        elif method == 'rank':
            # 排名标准化（百分位）
            return factor_values.rank(pct=True)

        else:
            return factor_values

    def winsorize(self, factor_values: pd.Series,
                  lower: float = 0.05,
                  upper: float = 0.95) -> pd.Series:
        """
        去极值处理

        Args:
            factor_values: 因子值
            lower: 下分位数
            upper: 上分位数

        Returns:
            去极值后的因子值
        """
        lower_bound = factor_values.quantile(lower)
        upper_bound = factor_values.quantile(upper)

        result = factor_values.copy()
        result[result < lower_bound] = lower_bound
        result[result > upper_bound] = upper_bound

        return result

    def neutralize(self, factor_values: pd.Series,
                   industry: Optional[pd.Series] = None,
                   market_cap: Optional[pd.Series] = None) -> pd.Series:
        """
        因子中性化（行业中性、市值中性）

        Args:
            factor_values: 因子值
            industry: 行业分类（索引需与factor_values相同）
            market_cap: 市值（索引需与factor_values相同）

        Returns:
            中性化后的因子值
        """
        if industry is None and market_cap is None:
            return factor_values

        df = pd.DataFrame({'factor': factor_values})

        if industry is not None:
            df['industry'] = industry
            # 行业中性：对每个行业内的因子进行标准化
            df['factor_industry_neutral'] = df.groupby('industry')['factor'].transform(
                lambda x: (x - x.mean()) / x.std()
            )

        if market_cap is not None:
            df['market_cap'] = market_cap
            # 市值中性：对市值进行回归，取残差
            from scipy import stats
            x = market_cap.values
            y = factor_values.values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            df['factor_market_neutral'] = y - (slope * x + intercept)

        # 返回处理后的因子
        if industry is not None and market_cap is not None:
            return df['factor_industry_neutral'] - df['factor_market_neutral']
        elif industry is not None:
            return df['factor_industry_neutral']
        elif market_cap is not None:
            return df['factor_market_neutral']
        else:
            return factor_values

    def preprocess(self, factor_values: pd.Series,
                   winsorize: bool = True,
                   standardize: bool = True,
                   handle_missing: bool = True) -> pd.Series:
        """
        因子预处理流程

        Args:
            factor_values: 原始因子值
            winsorize: 是否去极值
            standardize: 是否标准化
            handle_missing: 是否处理缺失值

        Returns:
            预处理后的因子值
        """
        result = factor_values.copy()

        if handle_missing:
            result = self.handle_missing_values(result)

        if winsorize:
            result = self.winsorize(result)

        if standardize:
            result = self.standardize(result)

        return result

    def get_info(self) -> Dict:
        """
        获取因子信息

        Returns:
            因子信息字典
        """
        return {
            'name': self.name,
            'params': self.params,
            'description': self.__doc__
        }

    def __repr__(self) -> str:
        return f"BaseFactor(name={self.name})"

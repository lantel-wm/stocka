"""
数据加载器
负责数据加载、标签计算、标准化等预处理操作
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np

from ..data.data_handler import DataHandler


class CSZScoreNorm:
    """
    Cross Sectional ZScore Normalization
    截面标准化：按日期分组，对每天的所有股票进行 z-score 标准化
    """

    def __init__(self, fields_group=None, method="zscore"):
        """
        初始化截面标准化器

        Args:
            fields_group: 字段分组（暂未使用）
            method: 标准化方法，"zscore" 或 "robust"
        """
        if method == "zscore":
            self.zscore_func = self._zscore
        elif method == "robust":
            self.zscore_func = self._robust_zscore
        else:
            raise ValueError(f"不支持的标准化方法: {method}")

    def __call__(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        对指定列进行截面标准化

        Args:
            df: 输入数据（需要有 date 列）
            columns: 需要标准化的列名列表

        Returns:
            标准化后的数据
        """
        # 按 date 分组，对每组（每天）进行 zscore
        df[columns] = df[columns].groupby("date", group_keys=False).apply(self.zscore_func)
        return df

    def _zscore(self, x: pd.Series) -> pd.Series:
        """标准 z-score 标准化"""
        return (x - x.mean()) / x.std()

    def _robust_zscore(self, x: pd.Series) -> pd.Series:
        """鲁棒 z-score 标准化（使用中位数和 MAD）"""
        median = x.median()
        mad = np.median(np.abs(x - median))
        return (x - median) / (mad * 1.4826)  # 1.4826 是使得 MAD 与标准差可比的常数


class DataLoader:
    """
    数据加载器

    功能：
    1. 加载指定日期范围的数据
    2. 计算标签（未来收益率）
    3. 截面标准化（Cross-Sectional Z-Score Normalization）
    4. 过滤极端标签值
    5. 填充缺失值

    参数：
        segment: 'train', 'valid', 'test'
        data_handler: 数据处理器
        factors: 因子列表
        start_date: 开始日期
        end_date: 结束日期
        label_threshold: 标签极端值阈值（默认0.12，过滤收益率>12%的样本）
        norm_method: 标准化方法，"zscore" 或 "robust"
    """

    def __init__(self,
                 segment: str,
                 data_handler: DataHandler,
                 factors: list,
                 start_date: str,
                 end_date: str,
                 label_threshold: float = 0.12,
                 norm_method: str = "zscore"):
        """
        初始化数据加载器

        Args:
            segment: 数据段类型 ('train', 'valid', 'test')
            data_handler: 数据处理器
            factors: 因子列表
            start_date: 开始日期
            end_date: 结束日期
            label_threshold: 标签极端值阈值
            norm_method: 标准化方法
        """
        assert segment in ['train', 'valid', 'test'], f"segment必须是 'train', 'valid', 'test' 之一"

        self.segment = segment
        self.data_handler = data_handler
        self.factors = factors
        self.start_date = start_date
        self.end_date = end_date
        self.label_threshold = label_threshold
        self.norm_method = norm_method

        # 创建截面标准化器
        self.cs_zscore_norm = CSZScoreNorm(method=norm_method)

        self.data = None

    def get_label(self, df: pd.DataFrame) -> pd.DataFrame:
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
        close_shift_1 = df.groupby(level=0)['close'].shift(-1)
        close_shift_2 = df.groupby(level=0)['close'].shift(-2)

        # 计算收益率
        df['LABEL0'] = close_shift_2 / close_shift_1 - 1

        return df

    def process_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理标签：过滤极端值

        Args:
            df: 包含LABEL0列的DataFrame

        Returns:
            处理后的DataFrame
        """
        # 过滤收益率超过阈值的样本
        df = df[df['LABEL0'].abs() <= self.label_threshold]
        # 删除标签为NaN的行
        df = df.dropna(subset=['LABEL0'])

        return df

    def load(self) -> pd.DataFrame:
        """
        加载并处理数据

        Returns:
            处理后的DataFrame
        """
        print(f"加载{self.segment}数据: {self.start_date} 至 {self.end_date}")

        # 获取数据（DataHandler已确保数据在指定时间范围内）
        data = self.data_handler.get_all_data()

        if data is None or len(data) == 0:
            raise ValueError("没有可用的数据")

        print(f"  - 原始样本数: {len(data)}")

        # 计算标签
        data = self.get_label(data)

        # 处理标签（过滤极端值）
        data = self.process_label(data)

        print(f"  - 过滤后样本数: {len(data)}")

        # 截面标准化（对 factors + LABEL0 进行标准化）
        cols_to_norm = self.factors + ['LABEL0']
        data = self.cs_zscore_norm(data, cols_to_norm)

        # 填充缺失值为0
        data[cols_to_norm] = data[cols_to_norm].fillna(0)

        print(f"  - 最终样本数: {len(data)}")

        self.data = data
        return data

    def get_data(self) -> pd.DataFrame:
        """获取加载的数据"""
        if self.data is None:
            raise ValueError("数据尚未加载，请先调用load()方法")
        return self.data

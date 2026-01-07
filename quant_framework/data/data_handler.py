"""
数据管理模块
负责加载、清洗和管理A股日线数据
"""

import os
import glob
from datetime import datetime, date
from typing import Optional, List, Dict
import pandas as pd
import numpy as np


class DataHandler:
    """
    数据处理器类
    从CSV文件加载A股日线数据，提供数据查询接口
    """

    # 列名映射（中文 -> 英文字段名）
    COLUMN_MAP = {
        '日期': 'date',
        '股票代码': 'code',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'amount',
        '振幅': 'amplitude',
        '涨跌幅': 'pct_change',
        '涨跌额': 'change_amount',
        '换手率': 'turnover'
    }

    # 反向映射（英文 -> 中文）
    COLUMN_MAP_REVERSE = {v: k for k, v in COLUMN_MAP.items()}

    def __init__(self, data_path: str, min_data_points: int = 100,
                 stock_whitelist: Optional[List[str]] = None):
        """
        初始化数据处理器

        Args:
            data_path: CSV数据文件所在目录
            min_data_points: 最少数据点数，用于过滤股票
            stock_whitelist: 股票白名单，只加载白名单中的股票（可选）
        """
        self.data_path = data_path
        self.min_data_points = min_data_points
        self.stock_whitelist = stock_whitelist  # 股票白名单
        self.all_data = None  # 存储所有数据的DataFrame
        self.available_dates = []  # 所有可用交易日期
        self.stock_codes = []  # 所有股票代码

    def load_data(self, start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> None:
        """
        加载CSV数据

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
        """
        # 检查数据目录是否存在
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据目录不存在: {self.data_path}")

        # 获取所有CSV文件
        all_files = glob.glob(os.path.join(self.data_path, "*.csv"))

        if not all_files:
            raise ValueError(f"数据目录中没有找到CSV文件: {self.data_path}")

        # 如果设置了白名单，过滤文件
        if self.stock_whitelist is not None:
            # 确保白名单中的代码是字符串类型
            whitelist = [str(code) for code in self.stock_whitelist]

            # 过滤CSV文件，只保留白名单中的股票
            filtered_files = []
            for file_path in all_files:
                code = os.path.basename(file_path).replace('.csv', '')
                if code in whitelist:
                    filtered_files.append(file_path)

            all_files = filtered_files

            if not all_files:
                raise ValueError(f"白名单中的股票在数据目录中未找到。白名单: {whitelist}")

            print(f"启用股票白名单，只加载 {len(all_files)} 只股票...")

        data_frames = []
        self.stock_codes = []

        print(f"开始加载数据，共找到 {len(all_files)} 个CSV文件...")

        for file_path in all_files:
            try:
                # 读取CSV文件，指定股票代码列为字符串类型以保留前导0
                df = pd.read_csv(file_path, encoding='utf-8', dtype={'股票代码': str})

                # 重命名列名为英文字段名
                df.rename(columns=self.COLUMN_MAP, inplace=True)

                # 如果文件中没有股票代码，从文件名提取（保留前导0）
                if 'code' not in df.columns or df['code'].isna().all():
                    code = os.path.basename(file_path).replace('.csv', '')
                    df['code'] = code
                else:
                    # 确保code列是字符串类型
                    df['code'] = df['code'].astype(str)

                # 确保日期格式正确
                df['date'] = pd.to_datetime(df['date'])

                # 过滤日期范围
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
                    
                # 过滤数据点不足的股票
                if len(df) >= self.min_data_points:
                    data_frames.append(df)
                    code = str(df['code'].iloc[0])  # 确保转换为字符串
                    if code not in self.stock_codes:
                        self.stock_codes.append(code)

            except Exception as e:
                print(f"读取文件失败 {file_path}: {e}")
                continue

        if not data_frames:
            raise ValueError("没有加载到任何有效数据")

        # 合并所有数据
        self.all_data = pd.concat(data_frames, ignore_index=True)

        # 确保code列是字符串类型（保留前导0）
        self.all_data['code'] = self.all_data['code'].astype(str)

        # 创建多级索引
        self.all_data.set_index(['code', 'date'], inplace=True)
        self.all_data.sort_index(inplace=True)

        # 获取所有可用交易日期
        self.available_dates = sorted(
            self.all_data.index.get_level_values('date').unique().to_list()
        )

        # 转换为date对象
        self.available_dates = [d.date() for d in self.available_dates]

        print(f"数据加载完成！")
        print(f"  - 股票数量: {len(self.stock_codes)}")
        print(f"  - 日期范围: {self.available_dates[0]} 至 {self.available_dates[-1]}")
        print(f"  - 总数据点: {len(self.all_data)}")

    def get_stock_data(self, code: str,
                       end_date: Optional[date] = None) -> pd.DataFrame:
        """
        获取单只股票的历史数据

        Args:
            code: 股票代码
            end_date: 截止日期（默认返回所有数据）

        Returns:
            股票历史数据DataFrame
        """
        if self.all_data is None:
            raise ValueError("数据未加载，请先调用 load_data()")

        try:
            if code in self.all_data.index:
                df = self.all_data.loc[code].copy()

                # 如果指定了截止日期，进行过滤
                if end_date:
                    df = df[df.index <= pd.Timestamp(end_date)]

                return df
            else:
                return pd.DataFrame()

        except KeyError:
            return pd.DataFrame()

    def get_data_before(self, code: str, date: date) -> pd.DataFrame:
        """
        获取指定日期之前的数据（避免未来函数）

        Args:
            code: 股票代码
            date: 截止日期

        Returns:
            历史数据DataFrame
        """
        df = self.get_stock_data(code)
        return df[df.index < pd.Timestamp(date)]

    def get_daily_data(self, date: date) -> pd.DataFrame:
        """
        获取指定日期的所有股票数据

        Args:
            date: 交易日期

        Returns:
            当日所有股票数据
        """
        if self.all_data is None:
            raise ValueError("数据未加载，请先调用 load_data()")

        try:
            return self.all_data.xs(pd.Timestamp(date), level='date')
        except KeyError:
            return pd.DataFrame()

    def get_multi_data(self, codes: List[str],
                       date: date) -> Dict[str, pd.Series]:
        """
        获取多只股票在指定日期的数据

        Args:
            codes: 股票代码列表
            date: 交易日期

        Returns:
            {股票代码: 当日数据Series} 的字典
        """
        result = {}
        daily_data = self.get_daily_data(date)

        for code in codes:
            if code in daily_data.index:
                result[code] = daily_data.loc[code]

        return result

    def get_all_codes(self) -> List[str]:
        """
        获取所有股票代码

        Returns:
            股票代码列表（字符串）
        """
        # 确保所有代码都是字符串类型
        return [str(code) for code in self.stock_codes]

    def get_available_dates(self, start_date: Optional[date] = None,
                            end_date: Optional[date] = None) -> List[date]:
        """
        获取可用交易日期列表

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日期列表
        """
        dates = self.available_dates

        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        return dates

    def update_data(self, new_data: pd.DataFrame) -> None:
        """
        更新数据（用于实盘）

        Args:
            new_data: 新的日线数据
        """
        if self.all_data is None:
            raise ValueError("基础数据未加载，请先调用 load_data()")

        # 确保列名正确
        new_data.rename(columns=self.COLUMN_MAP, inplace=True)

        # 确保日期格式
        new_data['date'] = pd.to_datetime(new_data['date'])

        # 添加到现有数据
        new_data.set_index(['code', 'date'], inplace=True)
        self.all_data = pd.concat([self.all_data, new_data])
        self.all_data.sort_index(inplace=True)

        # 更新可用日期
        self.available_dates = sorted(
            self.all_data.index.get_level_values('date').unique().to_list()
        )
        self.available_dates = [d.date() for d in self.available_dates]

    def get_data_info(self) -> Dict:
        """
        获取数据集信息

        Returns:
            数据集信息字典
        """
        if self.all_data is None:
            return {"status": "未加载数据"}

        return {
            "status": "已加载",
            "stock_count": len(self.stock_codes),
            "start_date": str(self.available_dates[0]),
            "end_date": str(self.available_dates[-1]),
            "trading_days": len(self.available_dates),
            "total_data_points": len(self.all_data)
        }

    def __repr__(self) -> str:
        """字符串表示"""
        info = self.get_data_info()
        if info["status"] == "未加载数据":
            return "DataHandler(未加载)"

        return (f"DataHandler(股票: {info['stock_count']}, "
                f"日期: {info['start_date']} 至 {info['end_date']}, "
                f"交易日: {info['trading_days']})")

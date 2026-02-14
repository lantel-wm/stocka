"""
数据管理模块（基于文件系统）
负责加载、清洗和管理A股日线数据
支持CSV和Parquet格式，支持多进程并行加载
"""

import os
import glob
from datetime import date
from typing import Optional, List, Dict
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

from quant_framework.utils.logger import get_logger

logger = get_logger(__name__)



class DataHandlerF:
    """
    数据处理器类（基于文件系统）
    从CSV/Parquet文件加载A股日线数据，提供数据查询接口
    支持自动格式检测和多进程并行加载
    """

    def __init__(self, data_path: str, min_data_points: int = 100,
                 stock_whitelist: Optional[List[str]] = None,
                 use_parquet: Optional[bool] = None,
                 num_workers: int = 1):
        """
        初始化数据处理器

        Args:
            data_path: 数据文件所在目录
            min_data_points: 最少数据点数，用于过滤股票
            stock_whitelist: 股票白名单，只加载白名单中的股票（可选）
            use_parquet: 是否使用Parquet格式（None=自动检测，True=强制Parquet，False=使用CSV）
            num_workers: 并行加载的进程数（1=单进程，>1=多进程，默认为1）
        """
        self.data_path = data_path
        self.min_data_points = min_data_points
        self.stock_whitelist = stock_whitelist
        self.use_parquet = use_parquet
        self.num_workers = num_workers if num_workers > 0 else 1
        self.all_data = None
        self.available_dates = []
        self.stock_codes = []

    def load_data(self, start_date: Optional[str] = None,
                  end_date: Optional[str] = None, factors: List[str] = None) -> None:
        """
        加载数据（支持CSV和Parquet格式，支持多进程并行）

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
        """
        # 检查数据目录是否存在
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据目录不存在: {self.data_path}")

        # 自动检测文件格式
        use_parquet = self._detect_format()

        # 确定文件扩展名
        file_ext = "*.parquet" if use_parquet else "*.csv"

        # 获取所有数据文件
        all_files = glob.glob(os.path.join(self.data_path, file_ext))

        if not all_files:
            raise ValueError(f"数据目录中没有找到数据文件: {self.data_path}")

        # 如果设置了白名单，过滤文件
        if self.stock_whitelist is not None:
            whitelist = [str(code) for code in self.stock_whitelist]
            filtered_files = []
            for file_path in all_files:
                code = os.path.basename(file_path).replace('.parquet', '').replace('.csv', '')
                if code in whitelist:
                    filtered_files.append(file_path)
            all_files = filtered_files

            if not all_files:
                raise ValueError(f"白名单中的股票在数据目录中未找到。白名单: {whitelist}")

            logger.info(f"启用股票白名单，只加载 {len(all_files)} 只股票...")

        # 显示加载信息
        format_name = "Parquet" if use_parquet else "CSV"
        logger.info(f"开始加载数据，共找到 {len(all_files)} 个{format_name}文件...")

        if self.num_workers > 1:
            logger.info(f"使用 {self.num_workers} 个进程并行加载...")
            data_frames = self._load_parallel(all_files, use_parquet, start_date, end_date, factors)
        else:
            data_frames = self._load_sequential(all_files, use_parquet, start_date, end_date, factors)

        if not data_frames:
            raise ValueError("没有加载到任何有效数据")

        # 合并所有数据
        logger.info("合并数据...")
        self.all_data = pd.concat(data_frames, ignore_index=True)

        # 确保code列是字符串类型
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

        logger.info(f"数据加载完成！")
        logger.info(f"  - 股票数量: {len(self.stock_codes)}")
        logger.info(f"  - 日期范围: {self.available_dates[0]} 至 {self.available_dates[-1]}")
        logger.info(f"  - 总数据点: {len(self.all_data)}")

    def _detect_format(self) -> bool:
        """自动检测使用哪种文件格式"""
        if self.use_parquet is not None:
            # 用户明确指定
            return self.use_parquet

        # 自动检测：优先使用Parquet
        parquet_files = glob.glob(os.path.join(self.data_path, "*.parquet"))
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))

        if parquet_files and not csv_files:
            logger.info("检测到Parquet文件，使用Parquet格式")
            return True
        elif csv_files and not parquet_files:
            logger.info("检测到CSV文件，使用CSV格式")
            return False
        elif parquet_files and csv_files:
            logger.info("同时检测到Parquet和CSV文件，优先使用Parquet格式")
            return True
        else:
            raise ValueError(f"数据目录中没有找到数据文件: {self.data_path}")

    def _load_sequential(self, all_files: List[str], use_parquet: bool,
                         start_date: Optional[str], end_date: Optional[str],
                         factors: List[str]) -> List[pd.DataFrame]:
        """单进程顺序加载数据"""
        data_frames = []
        self.stock_codes = []

        for file_path in tqdm(all_files, desc="加载进度"):
            df = self._load_single_file(file_path, use_parquet, start_date, end_date, factors)
            if df is not None and len(df) >= self.min_data_points:
                data_frames.append(df)
                code = str(df['code'].iloc[0])
                if code not in self.stock_codes:
                    self.stock_codes.append(code)

        return data_frames

    def _load_parallel(self, all_files: List[str], use_parquet: bool,
                       start_date: Optional[str], end_date: Optional[str],
                       factors: List[str]) -> List[pd.DataFrame]:
        """多进程并行加载数据"""
        data_frames = []
        self.stock_codes = []

        # 准备参数
        load_args = [(file_path, use_parquet, start_date, end_date, factors)
                     for file_path in all_files]

        with Pool(self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(self._load_single_file_wrapper, load_args),
                total=len(load_args),
                desc="加载进度"
            ))

        # 处理结果
        for df in results:
            if df is not None and len(df) >= self.min_data_points:
                data_frames.append(df)
                code = str(df['code'].iloc[0])
                if code not in self.stock_codes:
                    self.stock_codes.append(code)

        return data_frames

    @staticmethod
    def _load_single_file_wrapper(args):
        """包装函数用于多进程"""
        return DataHandlerF._load_single_file(*args)

    @staticmethod
    def _load_single_file(file_path: str, use_parquet: bool,
                          start_date: Optional[str], end_date: Optional[str],
                          factors: List[str]) -> Optional[pd.DataFrame]:
        """加载单个文件"""
        try:
            # 读取文件
            if use_parquet:
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, encoding='utf-8')

            # 如果文件中没有股票代码，从文件名提取（保留前导0）
            if 'code' not in df.columns or df['code'].isna().all():
                code = os.path.basename(file_path).replace('.parquet', '').replace('.csv', '')
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

            return df

        except Exception as e:
            logger.info(f"\n读取文件失败 {file_path}: {e}")
            return None

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
        return df[df.index <= pd.Timestamp(date)]

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
        获取多个股票在指定日期的数据

        Args:
            codes: 股票代码列表
            date: 交易日期

        Returns:
            字典，键为股票代码，值为该股票的数据（Series）
        """
        daily_data = self.get_daily_data(date)
        result = {}

        for code in codes:
            if code in daily_data.index:
                result[code] = daily_data.loc[code]

        return result

    def get_all_data(self) -> pd.DataFrame:
        """
        获取所有日期的所有股票数据

        Returns:
            所有股票数据
        """
        return self.all_data

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

    def get_previous_trading_date(self, current_date: date, n: int = 1) -> Optional[date]:
        """
        获取指定日期前n个交易日的日期

        Args:
            current_date: 当前日期
            n: 向前推的交易日数量（默认为1，即前一个交易日）

        Returns:
            前n个交易日的日期，如果找不到则返回None

        Example:
            >>> # 获取2024-01-10前一个交易日
            >>> prev_date = data_handler.get_previous_trading_date(date(2024, 1, 10))
            >>> # 获取2024-01-10前3个交易日
            >>> prev_3_date = data_handler.get_previous_trading_date(date(2024, 1, 10), n=3)
        """
        if self.all_data is None:
            raise ValueError("数据未加载，请先调用 load_data()")

        # 获取所有交易日期
        all_dates = self.available_dates

        # 找到current_date在交易日期中的位置
        try:
            current_idx = all_dates.index(current_date)
        except ValueError:
            # current_date不是交易日，找到不晚于current_date的最后一个交易日
            valid_dates = [d for d in all_dates if d <= current_date]
            if not valid_dates:
                return None
            current_idx = all_dates.index(valid_dates[-1])

        # 计算目标索引
        target_idx = current_idx - n

        # 检查索引是否有效
        if target_idx < 0:
            return None

        return all_dates[target_idx]

    def update_data(self, new_data: pd.DataFrame) -> None:
        """
        更新数据（用于实盘）

        Args:
            new_data: 新的日线数据
        """
        if self.all_data is None:
            raise ValueError("基础数据未加载，请先调用 load_data()")

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

    def load_index_data(self, index_path: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        加载指数数据（用于benchmark）

        Args:
            index_path: 指数数据文件路径（CSV格式，英文表头）
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）

        Returns:
            指数数据DataFrame，包含date和close列
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"指数数据文件不存在: {index_path}")

        # 读取指数数据
        index_df = pd.read_csv(index_path, encoding='utf-8')

        # 检查必要的列是否存在
        if 'date' not in index_df.columns or 'close' not in index_df.columns:
            raise ValueError(f"指数数据文件必须包含date和close列")

        # 转换日期格式
        index_df['date'] = pd.to_datetime(index_df['date'])

        # 过滤日期范围
        if start_date:
            index_df = index_df[index_df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            index_df = index_df[index_df['date'] <= pd.to_datetime(end_date)]

        # 只保留需要的列
        index_df = index_df[['date', 'close']].copy()

        # 按日期排序
        index_df = index_df.sort_values('date').reset_index(drop=True)

        return index_df

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
            return "DataHandlerF(未加载)"

        return (f"DataHandlerF(股票: {info['stock_count']}, "
                f"日期: {info['start_date']} 至 {info['end_date']}, "
                f"交易日: {info['trading_days']})")

    def close(self) -> None:
        """
        关闭连接（空实现，保持接口一致性）

        基于文件的实现不需要关闭连接
        """
        pass

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

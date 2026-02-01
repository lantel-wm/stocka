"""
数据更新器
负责从在线数据源（akshare）获取最新行情数据并增量更新本地文件
"""

import os
import time
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path


class DataUpdater:
    """
    数据更新器
    从 akshare 获取最新的 A 股行情数据，并增量更新到本地文件

    功能：
    - 自动检测现有数据的最新日期
    - 只下载新增的数据（增量更新）
    - 支持股票和指数数据更新
    - 支持批量更新和并发控制
    """

    # 列名映射（akshare -> 标准列名）
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

    def __init__(self,
                 output_dir: str,
                 use_parquet: bool = True,
                 delay: float = 0.5):
        """
        初始化数据更新器

        Args:
            output_dir: 数据文件输出目录
            use_parquet: 是否使用 Parquet 格式（True）或 CSV 格式（False）
            delay: 请求间隔时间（秒），避免频繁请求被限制
        """
        self.output_dir = output_dir
        self.use_parquet = use_parquet
        self.delay = delay

        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"✓ 数据更新器初始化完成")
        print(f"  - 输出目录: {output_dir}")
        print(f"  - 文件格式: {'Parquet' if use_parquet else 'CSV'}")
        print(f"  - 请求间隔: {delay} 秒")

    def update_stock_data(self,
                         stock_code: str,
                         end_date: Optional[str] = None) -> Dict:
        """
        更新单只股票的日线数据

        Args:
            stock_code: 股票代码（如 "600000" 或 "000001"）
            end_date: 结束日期（格式：YYYYMMDD，默认为今天）

        Returns:
            更新结果字典：
            {
                'stock_code': str,
                'status': 'success' | 'skipped' | 'error',
                'message': str,
                'existing_range': (start, end),  # 现有数据的日期范围
                'downloaded_range': (start, end),  # 下载的数据范围
                'new_rows': int,  # 新增的行数
                'total_rows': int  # 更新后的总行数
            }
        """
        # 格式化股票代码
        stock_code = str(stock_code).strip()

        # 设置结束日期
        if end_date is None:
            end_date = date.today().strftime('%Y%m%d')

        result = {
            'stock_code': stock_code,
            'status': 'error',
            'message': '',
            'existing_range': (None, None),
            'downloaded_range': (None, None),
            'new_rows': 0,
            'total_rows': 0
        }

        try:
            # 文件路径
            file_ext = '.parquet' if self.use_parquet else '.csv'
            file_path = os.path.join(self.output_dir, f"{stock_code}{file_ext}")

            # 获取现有数据的最新日期
            existing_start, existing_end = self._get_latest_date_from_file(file_path)
            result['existing_range'] = (existing_start, existing_end)

            # 确定下载数据的起始日期
            if existing_end is not None:
                # 如果有现有数据，从下一天开始下载
                start_date = (pd.to_datetime(existing_end) + timedelta(days=1)).strftime('%Y%m%d')
                # 如果起始日期已经超过结束日期，说明数据已经是最新的
                if start_date > end_date:
                    result['status'] = 'skipped'
                    result['message'] = f'数据已是最新（最新日期：{existing_end}）'
                    return result
            else:
                # 如果没有现有数据，下载全部历史数据（限制从2010年开始）
                start_date = '20100101'

            # 下载增量数据
            downloaded_data = self._fetch_stock_data_from_api(stock_code, start_date, end_date)

            if downloaded_data is None or len(downloaded_data) == 0:
                result['status'] = 'skipped'
                result['message'] = f'没有新数据需要下载（{start_date} 至 {end_date}）'
                return result

            # 更新结果中的下载范围
            downloaded_start = downloaded_data['date'].min().strftime('%Y-%m-%d')
            downloaded_end = downloaded_data['date'].max().strftime('%Y-%m-%d')
            result['downloaded_range'] = (downloaded_start, downloaded_end)
            result['new_rows'] = len(downloaded_data)

            # 保存数据
            if existing_end is not None and os.path.exists(file_path):
                # 读取现有数据并追加
                existing_data = self._read_data_file(file_path)
                combined_data = pd.concat([existing_data, downloaded_data], ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=['code', 'date'], keep='last')
                combined_data = combined_data.sort_values('date').reset_index(drop=True)
            else:
                # 直接保存新数据
                combined_data = downloaded_data

            # 写入文件
            self._write_data_file(file_path, combined_data)

            result['status'] = 'success'
            result['message'] = f'成功更新 {len(downloaded_data)} 条新数据'
            result['total_rows'] = len(combined_data)

            print(f"✓ {stock_code}: 更新成功 ({downloaded_start} 至 {downloaded_end}, +{len(downloaded_data)} 条)")
            exit(1)

        except Exception as e:
            result['status'] = 'error'
            result['message'] = f'更新失败: {str(e)}'
            print(f"✗ {stock_code}: {result['message']}")

        return result

    def update_batch_stock_data(self,
                               stock_codes: List[str],
                               end_date: Optional[str] = None,
                               delay: Optional[float] = None) -> Dict:
        """
        批量更新股票数据

        Args:
            stock_codes: 股票代码列表
            end_date: 结束日期（格式：YYYYMMDD，默认为今天）
            delay: 请求间隔时间（秒），默认使用初始化时的 delay

        Returns:
            批量更新结果字典：
            {
                'total': int,           # 总数
                'success': int,         # 成功数量
                'skipped': int,         # 跳过数量（已是最新）
                'error': int,           # 失败数量
                'details': List[Dict]   # 详细结果列表
            }
        """
        if delay is None:
            delay = self.delay

        results = {
            'total': len(stock_codes),
            'success': 0,
            'skipped': 0,
            'error': 0,
            'details': []
        }

        print(f"\n开始批量更新 {len(stock_codes)} 只股票...")

        for i, stock_code in enumerate(stock_codes):
            print(f"\n[{i+1}/{len(stock_codes)}] 更新 {stock_code}...")

            result = self.update_stock_data(stock_code, end_date)
            results['details'].append(result)

            # 统计
            if result['status'] == 'success':
                results['success'] += 1
            elif result['status'] == 'skipped':
                results['skipped'] += 1
            else:
                results['error'] += 1

            # 请求间隔（最后一次不需要等待）
            if i < len(stock_codes) - 1:
                time.sleep(delay)

        print(f"\n批量更新完成！")
        print(f"  总计: {results['total']} 只")
        print(f"  成功: {results['success']} 只")
        print(f"  跳过: {results['skipped']} 只")
        print(f"  失败: {results['error']} 只")

        return results

    def update_index_data(self,
                         index_code: str,
                         end_date: Optional[str] = None) -> Dict:
        """
        更新指数数据

        Args:
            index_code: 指数代码（如 "sh000300" 沪深300, "sz399006" 创业板指）
            end_date: 结束日期（格式：YYYYMMDD，默认为今天）

        Returns:
            更新结果字典（格式同 update_stock_data）
        """
        index_code = str(index_code).strip().lower()

        if end_date is None:
            end_date = date.today().strftime('%Y%m%d')

        result = {
            'stock_code': index_code,
            'status': 'error',
            'message': '',
            'existing_range': (None, None),
            'downloaded_range': (None, None),
            'new_rows': 0,
            'total_rows': 0
        }

        try:
            file_ext = '.parquet' if self.use_parquet else '.csv'
            file_path = os.path.join(self.output_dir, f"{index_code}{file_ext}")

            # 获取现有数据的最新日期
            existing_start, existing_end = self._get_latest_date_from_file(file_path)
            result['existing_range'] = (existing_start, existing_end)

            # 确定起始日期
            if existing_end is not None:
                start_date = (pd.to_datetime(existing_end) + timedelta(days=1)).strftime('%Y%m%d')
                if start_date > end_date:
                    result['status'] = 'skipped'
                    result['message'] = f'数据已是最新（最新日期：{existing_end}）'
                    return result
            else:
                start_date = '20100101'

            # 下载指数数据
            downloaded_data = self._fetch_index_data_from_api(index_code, start_date, end_date)

            if downloaded_data is None or len(downloaded_data) == 0:
                result['status'] = 'skipped'
                result['message'] = f'没有新数据需要下载（{start_date} 至 {end_date}）'
                return result

            downloaded_start = downloaded_data['date'].min().strftime('%Y-%m-%d')
            downloaded_end = downloaded_data['date'].max().strftime('%Y-%m-%d')
            result['downloaded_range'] = (downloaded_start, downloaded_end)
            result['new_rows'] = len(downloaded_data)

            # 保存数据
            if existing_end is not None and os.path.exists(file_path):
                existing_data = self._read_data_file(file_path)
                combined_data = pd.concat([existing_data, downloaded_data], ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=['date'], keep='last')
                combined_data = combined_data.sort_values('date').reset_index(drop=True)
            else:
                combined_data = downloaded_data

            self._write_data_file(file_path, combined_data)

            result['status'] = 'success'
            result['message'] = f'成功更新 {len(downloaded_data)} 条新数据'
            result['total_rows'] = len(combined_data)

            print(f"✓ {index_code}: 更新成功 ({downloaded_start} 至 {downloaded_end}, +{len(downloaded_data)} 条)")

        except Exception as e:
            result['status'] = 'error'
            result['message'] = f'更新失败: {str(e)}'
            print(f"✗ {index_code}: {result['message']}")

        return result

    def _get_latest_date_from_file(self, file_path: str) -> tuple:
        """
        从数据文件中获取日期范围

        Args:
            file_path: 文件路径

        Returns:
            (start_date, end_date) 日期元组，格式为 'YYYY-MM-DD' 或 None
        """
        if not os.path.exists(file_path):
            return (None, None)

        try:
            df = self._read_data_file(file_path)

            if df is None or len(df) == 0:
                return (None, None)

            # 获取日期范围
            start_date = df['date'].min().strftime('%Y-%m-%d')
            end_date = df['date'].max().strftime('%Y-%m-%d')

            return (start_date, end_date)

        except Exception as e:
            print(f"读取文件日期范围失败 {file_path}: {e}")
            return (None, None)

    def _read_data_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        读取数据文件

        Args:
            file_path: 文件路径

        Returns:
            DataFrame 或 None
        """
        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, encoding='utf-8')

            # 如果文件中是中文列名，转换为英文（使用 COLUMN_MAP）
            if '日期' in df.columns:
                df = df.rename(columns=self.COLUMN_MAP)

            # 确保日期格式
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            return df

        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None

    def _write_data_file(self, file_path: str, df: pd.DataFrame) -> None:
        """
        写入数据文件

        Args:
            file_path: 文件路径
            df: 要保存的 DataFrame
        """
        try:
            # 保存为 CSV 时使用中文列名（与原始数据格式一致）
            if file_path.endswith('.csv'):
                # 将英文列名转换回中文
                reverse_map = {v: k for k, v in self.COLUMN_MAP.items()}
                df_to_save = df.rename(columns=reverse_map)
                df_to_save.to_csv(file_path, index=False, encoding='utf-8')
            else:
                # Parquet 保持英文列名
                df.to_parquet(file_path, index=False)
        except Exception as e:
            raise IOError(f"写入文件失败 {file_path}: {e}")

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
    def _fetch_stock_data_from_api(self,
                                   stock_code: str,
                                   start_date: str,
                                   end_date: str) -> Optional[pd.DataFrame]:
        """
        从 akshare API 获取股票数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            DataFrame 或 None
        """
        try:
            import akshare as ak

            # 在调用 API 之前 sleep，防止触发使用限制
            time.sleep(0.5)

            print(f"  正在从 akshare 下载数据 ({start_date} 至 {end_date})...")

            # 获取股票历史数据
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="hfq"  # 后复权
            )

            if df is None or len(df) == 0:
                return None

            # 使用 COLUMN_MAP 重命名列（与 DataHandler 保持一致）
            df = df.rename(columns=self.COLUMN_MAP)

            # 添加股票代码列
            df['code'] = stock_code

            # 选择需要的列（按 COLUMN_MAP 中定义的英文字段名）
            required_columns = [
                'date', 'code', 'open', 'close', 'high', 'low',
                'volume', 'amount', 'amplitude', 'pct_change',
                'change_amount', 'turnover'
            ]

            # 只保留存在的列
            available_columns = [col for col in required_columns if col in df.columns]
            df = df[available_columns]

            # 确保日期格式
            df['date'] = pd.to_datetime(df['date'])

            return df

        except ImportError:
            print(f"  导入错误: 需要安装 akshare 库（pip install akshare）")
            return None
        except Exception as e:
            # 让异常传播到 retry 装饰器
            raise  # 重新抛出异常，让 retry 处理

    def _fetch_index_data_from_api(self,
                                   index_code: str,
                                   start_date: str,
                                   end_date: str) -> Optional[pd.DataFrame]:
        """
        从 akshare API 获取指数数据

        Args:
            index_code: 指数代码（如 sh000300, sz399006）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            DataFrame 或 None
        """
        try:
            import akshare as ak

            # 在调用 API 之前 sleep，防止触发使用限制
            time.sleep(0.5)

            print(f"  正在从 akshare 下载指数数据 ({start_date} 至 {end_date})...")

            # 获取指数历史数据
            df = ak.index_zh_a_hist(
                symbol=index_code,
                period="daily",
                start_date=start_date,
                end_date=end_date
            )

            if df is None or len(df) == 0:
                return None

            # 使用 COLUMN_MAP 重命名列（与 DataHandler 保持一致）
            df = df.rename(columns=self.COLUMN_MAP)

            # 选择需要的列（按 COLUMN_MAP 中定义的英文字段名）
            required_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']
            available_columns = [col for col in required_columns if col in df.columns]
            df = df[available_columns]

            # 确保日期格式
            df['date'] = pd.to_datetime(df['date'])

            return df

        except ImportError:
            print(f"  导入错误: 需要安装 akshare 库（pip install akshare）")
            return None
        except Exception as e:
            # 让异常传播到 retry 装饰器
            raise  # 重新抛出异常，让 retry 处理

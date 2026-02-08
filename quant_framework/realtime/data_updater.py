"""
数据更新器
负责从在线数据源（akshare）获取最新行情数据并增量更新到数据库
"""

import time
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List
from ..utils.logger import get_logger
from ..data.data_handler import DataHandler

logger = get_logger(__name__)


class DataUpdater:
    """
    数据更新器
    从 akshare 获取最新的 A 股行情数据，并增量更新到数据库

    功能：
    - 自动检测现有数据的最新日期
    - 只下载新增的数据（增量更新）
    - 支持批量更新
    - 使用 DataHandler 进行数据存储
    """

    def __init__(self, data_handler: DataHandler, delay: float = 0.5):
        """
        初始化数据更新器

        Args:
            data_handler: DataHandler 实例，用于数据存储和查询
            delay: 请求间隔时间（秒），避免频繁请求被限制
        """
        self.handler = data_handler
        self.delay = delay

        logger.info(f"✓ 数据更新器初始化完成")
        logger.info(f"  - 数据库: {self.handler.db_path}")
        logger.info(f"  - 请求间隔: {delay} 秒")

    def get_appropriate_end_date(self, target_date: Optional[date] = None) -> date:
        """
        根据当前时间和交易日历判断应该使用的结束日期

        时间判断逻辑：
        - A股市场在每个交易日 15:00 收盘
        - 当天的历史行情数据需要在收盘后才能从数据源获取
        - 设置 16:00 为分界点是为了增加鲁棒性，给数据源留出1小时缓冲时间
        - 如果目标日期不是交易日，找到最近的交易日
        - 如果目标日期是交易日但时间 < 16:00，使用前一个交易日
        - 如果目标日期是交易日且时间 >= 16:00，使用当天的数据

        Args:
            target_date: 目标日期（默认为今天）

        Returns:
            应该使用的结束日期
        """
        if target_date is None:
            target_date = date.today()

        # 获取当前时间
        now = datetime.now()

        # 1. 判断目标日期是否为交易日
        if self.handler.is_trading_day(target_date):
            # 是交易日
            if now.hour < 16:
                # 未到16:00，使用前一个交易日
                appropriate_date = self.handler.get_previous_trading_date(target_date, n=1)
                if appropriate_date is None:
                    # 如果无法获取前一个交易日，回退到简单减1
                    appropriate_date = target_date - timedelta(days=1)
                    logger.warning(f"无法获取前一个交易日，使用简单日期减1: {appropriate_date}")
                else:
                    logger.info(f"当前为交易日 {target_date}，时间 {now.strftime('%H:%M')} < 16:00，使用前一个交易日: {appropriate_date}")
            else:
                # 已到16:00，使用当天
                appropriate_date = target_date
                logger.info(f"当前为交易日 {target_date}，时间 {now.strftime('%H:%M')} >= 16:00，使用当天数据: {appropriate_date}")
        else:
            # 不是交易日，找到最近的交易日（向前查找）
            appropriate_date = self.handler.get_previous_trading_date(target_date, n=0)
            if appropriate_date is None:
                # 如果找不到前一个交易日，回退到简单减1
                appropriate_date = target_date - timedelta(days=1)
                logger.warning(f"{target_date} 不是交易日且无法找到前一个交易日，使用简单日期减1: {appropriate_date}")
            else:
                logger.info(f"{target_date} 不是交易日，使用最近的交易日: {appropriate_date}")

        return appropriate_date

    def update_stock_data(self,
                         stock_code: str,
                         end_date: Optional[str] = None) -> Dict:
        """
        更新单只股票的日线数据

        Args:
            stock_code: 股票代码（如 "600000" 或 "000001"）
            end_date: 结束日期（格式：YYYYMMDD，默认为今天）
                      如果为今天，会根据当前时间自动判断：
                      - 16:00 之前使用前一天
                      - 16:00 之后使用当天

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

        # 设置结束日期（根据时间自动判断）
        if end_date is None:
            appropriate_date = self.get_appropriate_end_date()
            end_date = appropriate_date.strftime('%Y%m%d')
        else:
            # 检查传入的日期是否是今天或未来日期
            try:
                input_date = datetime.strptime(end_date, '%Y%m%d').date()
                today = date.today()
                if input_date >= today:
                    # 如果是今天或未来日期，使用时间判断逻辑
                    appropriate_date = self.get_appropriate_end_date(input_date)
                    end_date = appropriate_date.strftime('%Y%m%d')
            except ValueError:
                pass  # 日期格式错误，让后续逻辑处理

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
            # 获取现有数据的最新日期
            existing_start, existing_end = self.handler.get_stock_latest_date(stock_code)

            # 转换为字符串格式以保持一致性
            existing_start_str = existing_start.strftime('%Y-%m-%d') if existing_start else None
            existing_end_str = existing_end.strftime('%Y-%m-%d') if existing_end else None
            result['existing_range'] = (existing_start_str, existing_end_str)

            # 确定下载数据的起始日期
            if existing_end is not None:
                # 如果有现有数据，从下一天开始下载
                start_date = (existing_end + timedelta(days=1)).strftime('%Y%m%d')
                # 如果起始日期已经超过结束日期，说明数据已经是最新的
                if start_date > end_date:
                    result['status'] = 'skipped'
                    result['message'] = f'数据已是最新（最新日期：{existing_end_str}）'
                    logger.info(f"⊘ {stock_code}: 跳过（数据已是最新，最新日期：{existing_end_str}）")
                    return result
            else:
                # 如果没有现有数据，下载全部历史数据（限制从2005年开始）
                start_date = '20050101'

            # 下载增量数据
            downloaded_data = self._fetch_stock_data_from_api(stock_code, start_date, end_date)
            downloaded_data['code'] = stock_code

            if downloaded_data is None or len(downloaded_data) == 0:
                result['status'] = 'skipped'
                result['message'] = f'没有新数据需要下载（{start_date} 至 {end_date}）'
                logger.info(f"⊘ {stock_code}: 跳过（{start_date} 至 {end_date} 暂无数据）")
                return result

            # 更新结果中的下载范围
            downloaded_start = downloaded_data['date'].min().strftime('%Y-%m-%d')
            downloaded_end = downloaded_data['date'].max().strftime('%Y-%m-%d')
            result['downloaded_range'] = (downloaded_start, downloaded_end)
            result['new_rows'] = len(downloaded_data)

            # 更新到数据库（DataHandler 会自动处理去重和排序）
            self.handler.update_data(downloaded_data)

            # 获取更新后的总行数
            updated_df = self.handler.get_stock_data(stock_code)
            result['total_rows'] = len(updated_df)

            result['status'] = 'success'
            result['message'] = f'成功更新 {len(downloaded_data)} 条新数据'

            logger.info(f"✓ {stock_code}: 更新成功 ({downloaded_start} 至 {downloaded_end}, +{len(downloaded_data)} 条)")

        except Exception as e:
            result['status'] = 'error'
            result['message'] = f'更新失败: {str(e)}'
            logger.error(f"✗ {stock_code}: {result['message']}")

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
                      如果为今天，会根据当前时间自动判断：
                      - 16:00 之前使用前一天
                      - 16:00 之后使用当天
            delay: 请求间隔时间（秒），默认使用初始化时的 delay

        Returns:
            批量更新结果字典：
            {
                'total': int,           # 总数
                'success': int,         # 成功数量
                'skipped': int,         # 跳过数量（已是最新）
                'error': int,           # 失败数量
                'failed_stocks': List[str]  # 更新失败的股票代码列表
                'details': List[Dict]   # 详细结果列表
            }
        """
        if delay is None:
            delay = self.delay

        # 如果 end_date 为空或为今天，根据当前时间判断应该使用的日期
        if end_date is None:
            appropriate_date = self.get_appropriate_end_date()
            end_date = appropriate_date.strftime('%Y%m%d')
        else:
            # 检查传入的日期是否是今天
            try:
                input_date = datetime.strptime(end_date, '%Y%m%d').date()
                today = date.today()
                if input_date >= today:
                    # 如果是今天或未来日期，使用时间判断逻辑
                    appropriate_date = self.get_appropriate_end_date(input_date)
                    end_date = appropriate_date.strftime('%Y%m%d')
            except ValueError:
                pass  # 日期格式错误，让后续逻辑处理

        results = {
            'total': len(stock_codes),
            'success': 0,
            'skipped': 0,
            'error': 0,
            'failed_stocks': [],
            'details': []
        }

        logger.info(f"\n开始批量更新 {len(stock_codes)} 只股票...")

        for i, stock_code in enumerate(stock_codes):
            logger.info(f"[{i+1}/{len(stock_codes)}] 更新 {stock_code}...")

            result = self.update_stock_data(stock_code, end_date)
            results['details'].append(result)

            # 统计
            if result['status'] == 'success':
                results['success'] += 1
            elif result['status'] == 'skipped':
                results['skipped'] += 1
            else:
                results['error'] += 1
                results['failed_stocks'].append(stock_code)

        logger.info(f"\n批量更新完成！")
        logger.info(f"  总计: {results['total']} 只")
        logger.info(f"  成功: {results['success']} 只")
        logger.info(f"  跳过: {results['skipped']} 只")
        logger.info(f"  失败: {results['error']} 只")

        return results

    # @retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
    @retry(stop=stop_after_attempt(1), wait=wait_fixed(1))
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
            time.sleep(self.delay)

            # 获取股票历史数据
            prefix = 'sh' if stock_code.startswith('6') else 'sz'
            logger.info(f"symbol={prefix}{stock_code}, start_date={start_date}, end_date={end_date}")
            df = ak.stock_zh_a_daily(
                symbol=f"{prefix}{stock_code}",
                start_date=start_date,
                end_date=end_date,
                adjust="hfq"
            )

            if df is None or len(df) == 0:
                return None

            # 确保日期格式
            df['date'] = pd.to_datetime(df['date'])

            return df

        except ImportError:
            logger.error(f"  导入错误: 需要安装 akshare 库（pip install akshare）")
            return None
        except Exception as e:
            # 让异常传播到 retry 装饰器
            raise  # 重新抛出异常，让 retry 处理

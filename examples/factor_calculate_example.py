import sys
import os

# STOCKA_BASE_DIR = '/Users/zhaozhiyu/Projects/stocka'
STOCKA_BASE_DIR = '/home/zzy/projects/stocka'
sys.path.insert(0, STOCKA_BASE_DIR)

from quant_framework import (
    DataHandler,
    Alpha158
)
from quant_framework.utils.logger import get_logger

logger = get_logger(__name__)

import pandas as pd
from tqdm import tqdm

import multiprocessing as mp
from functools import partial

def _process_single_stock(code, db_path):
    """
    处理单个股票的因子计算（工作函数）

    Args:
        code: 股票代码
        db_path: 数据库路径

    Returns:
        tuple: (code, factor_df) 或 (code, None) 如果失败
    """
    try:
        # 在每个进程中创建新的 DataHandler 实例
        from quant_framework import DataHandler, Alpha158

        data_handler = DataHandler(db_path)

        # 获取该股票的所有历史数据
        df = data_handler.get_stock_data(
            code,
            end_date="2026-02-05"
        )

        if len(df) > 0 and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # 确保数据类型正确
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

            # 计算因子（返回包含原始数据和所有因子的DataFrame）
            alpha158 = Alpha158()
            factor_df = alpha158.calculate(df)

            return (code, factor_df)

        return (code, None)

    except Exception as e:
        # 静默处理错误
        return (code, None)


def calculate_factor_values(data_handler, codes, db_path, num_workers=None):
    """
    计算指定因子的值（支持多核并行）

    Args:
        data_handler: 数据处理器
        codes: 股票代码列表
        db_path: 数据库路径
        num_workers: cpu数（默认使用所有可用核心）

    Returns:
        int: 成功计算的股票数量
    """
    # 确定使用的CPU核心数
    if num_workers is None:
        num_workers = mp.cpu_count()

    logger.info(f"使用 {num_workers} 个 CPU 核心进行并行计算...")
    logger.info(f"待计算股票数量: {len(codes)}")

    # 准备工作函数的固定参数
    worker_func = partial(
        _process_single_stock,
        db_path=db_path
    )

    # 使用进程池并行计算
    all_factor_dfs = []

    # 使用 imap_unordered 以获得流式结果
    with mp.Pool(processes=num_workers) as pool:
        # 使用 tqdm 显示进度
        results = list(tqdm(
            pool.imap_unordered(worker_func, codes),
            total=len(codes),
            desc="计算因子进度"
        ))

    # 整理结果：收集所有成功的因子DataFrame
    success_count = 0
    for code, factor_df in results:
        if factor_df is not None:
            all_factor_dfs.append((code, factor_df))
            success_count += 1

    logger.info(f"计算完成！成功: {success_count}/{len(codes)}")

    # 按日期组织并保存因子数据到数据库
    logger.info("开始保存因子数据到数据库...")
    save_factors_by_date(data_handler, all_factor_dfs)

    return success_count


def save_factors_by_date(data_handler, all_factor_dfs):
    """
    将所有股票的因子数据按日期保存到数据库

    Args:
        data_handler: 数据处理器
        all_factor_dfs: list of (code, factor_df) tuples
    """
    from datetime import date
    from collections import defaultdict

    # 收集所有日期和对应的股票因子
    date_factors = defaultdict(dict)  # {date: {code: factor_values}}

    for code, factor_df in all_factor_dfs:
        if factor_df is None or len(factor_df) == 0:
            continue

        # 检查是否有 'date' 列
        if 'date' in factor_df.columns:
            # 使用 'date' 列作为日期
            for _, row in factor_df.iterrows():
                trade_date = row['date']
                if hasattr(trade_date, 'date'):  # 如果是Timestamp
                    trade_date = trade_date.date()

                # 提取所有因子列（非原始数据列）
                # Alpha158因子列名通常是大写字母开头的
                factor_cols = [col for col in factor_df.columns
                              if col not in ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
                              and not col.startswith('Unnamed')]

                if factor_cols:
                    date_factors[trade_date][code] = row[factor_cols]
        else:
            # 使用索引作为日期
            for idx, row in factor_df.iterrows():
                if hasattr(idx, 'date'):  # 如果是DatetimeIndex
                    trade_date = idx.date()
                else:
                    trade_date = idx

                # 提取所有因子列（非原始数据列）
                factor_cols = [col for col in factor_df.columns
                              if col not in ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
                              and not col.startswith('Unnamed')]

                if factor_cols:
                    date_factors[trade_date][code] = row[factor_cols]

    # 按日期保存到数据库
    saved_dates = 0
    for trade_date, stock_factors in sorted(date_factors.items()):
        # 构建该日期的截面因子DataFrame
        factor_df = pd.DataFrame.from_dict(stock_factors, orient='index')
        factor_df.index.name = 'stock_code'

        # 保存到数据库
        try:
            data_handler.save_factors(factor_df, trade_date)
            saved_dates += 1
        except Exception as e:
            logger.warning(f"保存日期 {trade_date} 的因子失败: {e}")

    logger.info(f"成功保存 {saved_dates} 个交易日的因子数据到数据库")

if __name__ == '__main__':
    # 数据库路径
    db_path = os.path.join(STOCKA_BASE_DIR, "data/stock.db")

    # 创建DataHandler实例
    data_handler = DataHandler(db_path)

    # 初始化因子表
    logger.info("初始化因子表...")
    data_handler._init_factor_tables()
    logger.info("因子表初始化完成")
    logger.info("")

    # 获取所有股票代码
    all_codes = data_handler.get_all_codes()
    logger.info(f"数据库中股票数量: {len(all_codes)}")
    logger.info("")

    # 计算Alpha158因子
    logger.info("开始计算Alpha158因子...")
    success_count = calculate_factor_values(
        data_handler,
        all_codes,  # 使用所有股票
        db_path,
        num_workers=32,
    )

    logger.info("")
    logger.info(f"所有因子计算完成！成功处理: {success_count} 只股票")
    logger.info("")
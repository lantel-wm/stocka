import sys
import os

STOCKA_BASE_DIR = '/Users/zhaozhiyu/Projects/stocka'
# STOCKA_BASE_DIR = '/home/zzy/projects/stocka'
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

def _process_single_stock(code, data_handler):
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
        from quant_framework import Alpha158

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


def calculate_factor_values(data_handler, codes, parquet_save_dir):
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
    logger.info(f"待计算股票数量: {len(codes)}")

    for code in tqdm(codes):
        code, factor_df = _process_single_stock(code, data_handler)
        if factor_df is None:
            continue
        
        factor_df.set_index('code', inplace=True)
        
        parquet_save_path = os.path.join(parquet_save_dir, f"{code}.parquet")
        factor_df.to_parquet(parquet_save_path)
            

if __name__ == '__main__':
    # 数据库路径
    db_path = os.path.join(STOCKA_BASE_DIR, "data/stock.db")
    parquet_save_dir = os.path.join(STOCKA_BASE_DIR, "data/factor/day/alpha158")
    
    if not os.path.exists(parquet_save_dir):
        os.makedirs(parquet_save_dir)

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
    calculate_factor_values(
        data_handler,
        all_codes,  # 使用所有股票
        parquet_save_dir,
    )
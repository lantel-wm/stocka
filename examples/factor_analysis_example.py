import sys
import os

STOCKA_BASE_DIR = '/Users/zhaozhiyu/Projects/stocka'
sys.path.insert(0, STOCKA_BASE_DIR)

from quant_framework import (
    DataHandler,
    FactorMetrics,
    MultiFactorAnalysis,
    Alpha158
)

import pandas as pd
import numpy as np
from tqdm import tqdm

import multiprocessing as mp
from functools import partial

def _process_single_stock(code, data_path, min_data_points, date, save_dir):
    """
    处理单个股票的因子计算（工作函数）
    
    Args:
        code: 股票代码
        data_path: 数据路径
        min_data_points: 最小数据点数
        date: 计算日期
        save_dir: 保存目录
    
    Returns:
        tuple: (code, factor_df) 或 (code, None) 如果失败
    """
    try:
        # 在每个进程中创建新的 DataHandler 实例
        from quant_framework import DataHandler, Alpha158
        
        data_handler = DataHandler(
            data_path=data_path,
            min_data_points=min_data_points,
            stock_whitelist=[code]
        )
        
        data_handler.load_data(
            start_date="2015-01-05",
            end_date="2025-12-31"
        )
        
        # 获取该股票在指定日期之前的所有历史数据
        df = data_handler.get_data_before(code, date)

        if len(df) > 0 and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # 确保数据类型正确
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

            # 计算因子
            alpha158 = Alpha158()
            factor_df = alpha158.calculate(df)
            
            # 如果指定了保存目录，保存到文件
            if save_dir is not None:
                save_path = os.path.join(save_dir, f"{code}.csv")
                factor_df.to_csv(save_path)
                return (code, None)  # 保存到文件，不返回数据
            
            return (code, factor_df)
        
        return (code, None)
    
    except Exception as e:
        # 静默处理错误
        return (code, None)


def calculate_factor_values(data_handler, codes, dates, save_dir=None, num_workers=None):
    """
    计算指定因子的值（支持多核并行）

    Args:
        data_handler: 数据处理器
        codes: 股票代码列表
        dates: 日期列表
        save_dir: 保存路径
        num_workers: cpu数（默认使用所有可用核心）

    Returns:
        Dict[str, DataFrame]: key为股票代码, value为DataFrame
    """
    # 创建保存目录
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 获取计算日期
    date = dates[-1]
    
    # 确定使用的CPU核心数
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print(f"使用 {num_workers} 个 CPU 核心进行并行计算...")
    print(f"待计算股票数量: {len(codes)}")
    
    # 准备工作函数的固定参数
    worker_func = partial(
        _process_single_stock,
        data_path=data_handler.data_path,
        min_data_points=data_handler.min_data_points,
        date=date,
        save_dir=save_dir
    )
    
    # 使用进程池并行计算
    factor_dfs = {}
    
    # 使用 imap_unordered 以获得流式结果
    with mp.Pool(processes=num_workers) as pool:
        # 使用 tqdm 显示进度
        results = list(tqdm(
            pool.imap_unordered(worker_func, codes),
            total=len(codes),
            desc="计算因子进度"
        ))
    
    # 整理结果
    success_count = 0
    for code, factor_df in results:
        if factor_df is not None:
            factor_dfs[code] = factor_df
            success_count += 1
    
    print(f"计算完成！成功: {success_count}/{len(codes)}")
    
    return factor_dfs

if __name__ == '__main__':
    # # 测试50只股票
    # stock_list = [
    #     '000001', '000002', '000063', '000069', '000100', '000157', '000166', '000333', '000338', '000581',
    #     '000651', '000725', '000768', '000776', '000858', '000876', '000895', '000938', '000999', '001979',
    #     '002001', '002008', '002027', '002032', '002044', '002050', '002142', '002304', '002415', '002456',
    #     '002475', '002594', '600000', '600036', '600519', '600900', '601012', '601066', '601138', '601166',
    #     '601288', '601318', '601398', '601601', '601628', '601766', '601857', '601988', '603259', '688981'
    # ]

    # 测试1只股票
    stock_list = [
        '000001',
    ]

    data_handler = DataHandler(
        data_path=os.path.join(STOCKA_BASE_DIR, "data/stock/kline/day"),
        min_data_points=100,
        # stock_whitelist=stock_list
    )

    try:
        data_handler.load_data(
            start_date="2015-01-05",
            end_date="2025-12-31"
        )
        print(f"数据加载成功，股票数量: {len(data_handler.get_all_codes())}")
        print()

    except Exception as e:
        print(f"加载数据时出错：{e}")
        
    # 获取交易日期
    dates = data_handler.get_available_dates()
    dates = [d for d in dates if pd.to_datetime('2015-01-05').to_pydatetime().date() <= d <= pd.to_datetime('2025-12-31').to_pydatetime().date()]

    print(f"分析时间范围: {dates[0]} 至 {dates[-1]}")
    print(f"交易日数: {len(dates)}")
        
    # 计算所有因子
    print(f"计算Alpha158因子...")
    # factor_dfs = calculate_factor_values(data_handler, stock_list, dates)
    factor_dfs = calculate_factor_values(
        data_handler, 
        data_handler.get_all_codes(), 
        dates, 
        os.path.join(STOCKA_BASE_DIR, 'data/factor/day/alpha158'),
        num_workers=10,
    )

    print()
    print("所有因子计算完成！")
    print()
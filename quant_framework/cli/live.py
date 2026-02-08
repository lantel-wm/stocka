import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd


from datetime import date
from tqdm import tqdm
from typing import Optional

from quant_framework import MLStrategy, DataHandler, DataUpdater, Alpha158
from quant_framework.utils.logger import get_logger

# # 设置环境变量，让所有模块的日志都输出到同一个文件
# os.environ['STOCKA_LOG_FILE'] = 'logs/realtime_example.log'

logger = get_logger(__name__)

# STOCKA_BASE_DIR = "/home/zzy/projects/stocka"
STOCKA_BASE_DIR = "/Users/zhaozhiyu/Projects/stocka"

def update_stock(db_path: str, target_date: Optional[date] = None):
    data_handler = DataHandler(db_path)
    data_updater = DataUpdater(data_handler)
    
    if target_date is None:
        target_date = data_updater.get_appropriate_end_date()
        
    stock_code_list = data_handler.get_all_codes()
    
    update_result = data_updater.update_batch_stock_data(
        stock_codes=stock_code_list, end_date=target_date.strftime('%Y%m%d')
    )
    
    update_failure_stocks = update_result['failed_stocks']
    logger.info(f"更新失败列表: {update_failure_stocks}")
    
def update_factor()
        
    
        
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd


from datetime import date
from quant_framework import MLStrategy, DataHandler, DataUpdater, Alpha158
from quant_framework.utils.logger import get_logger
from tqdm import tqdm

# # 设置环境变量，让所有模块的日志都输出到同一个文件
# os.environ['STOCKA_LOG_FILE'] = 'logs/realtime_example.log'

logger = get_logger(__name__)

# STOCKA_BASE_DIR = "/home/zzy/projects/stocka"
STOCKA_BASE_DIR = "/Users/zhaozhiyu/Projects/stocka"

def update_factors(
    data_handler: DataHandler,
    data_updater: DataUpdater,
    last_trading_date: date,
    factor_calculator
):
    

    stock_code_list = data_handler.get_all_codes()
    # stock_code_list = ["000001", "000002", "600000"]

    # 1. 更新价格数据
    update_result = data_updater.update_batch_stock_data(
        stock_codes=stock_code_list,
        end_date=last_trading_date.strftime('%Y%m%d'),
        delay=0.3,
    )
    
    update_failure_stocks = update_result['failed_stocks']
    logger.info(f"更新失败列表: {update_failure_stocks}")

    # 2. 注册因子定义
    for factor_id, factor_name, factor_category, factor_desc in Alpha158.DEFINITIONS:
        data_handler.register_factor(factor_id, factor_name, factor_category, factor_desc)

    factor_end_date = last_trading_date
    factor_start_date = data_handler.get_previous_trading_date(factor_end_date, 60)

    factor_df = None
    skipped_count = 0
    calculated_count = 0

    pbar = tqdm(stock_code_list)
    logger.info("开始计算因子")
    for code in pbar:
        pbar.set_postfix({"code": code, "skip": skipped_count, "calc": calculated_count})
        
        if code in update_failure_stocks:
            skipped_count += 1
            continue

        # 3. 检查是否已存在该交易日的因子记录
        existing_factors = data_handler.get_stock_factors(
            stock_code=code,
            start_date=factor_end_date,
            end_date=factor_end_date
        )
        
        if not existing_factors.empty:
            # 已存在因子记录，跳过计算
            skipped_count += 1
            # logger.debug(f"股票 {code} 在 {factor_end_date} 的因子已存在，跳过计算")
            continue

        # 4. 计算新因子
        df = data_handler.get_range_data([code], factor_start_date, factor_end_date)
        
        new_factor_row = factor_calculator.calculate(df).iloc[[-1]]

        if factor_df is None:
            factor_df = new_factor_row
        else:
            factor_df = pd.concat([factor_df, new_factor_row], ignore_index=False)
            
        calculated_count += 1
        
    if factor_df is not None:
        factor_df = factor_df.set_index('code')


    logger.info(f"因子计算完成: 共 {len(stock_code_list)} 只股票，"
                f"跳过 {skipped_count} 只，新计算 {calculated_count} 只")
    
    # 5. 因子入库
    if factor_df is not None:
        factor_names = [d[1] for d in Alpha158.DEFINITIONS]
        data_handler.save_factors(factor_df[factor_names], factor_end_date)
    
    
def get_top_n_stocks(data_handler: DataHandler, last_trading_date: date):
    strategy = MLStrategy({
        # 'model_path': os.path.join(STOCKA_BASE_DIR, './ckpt/lightgbm_model_2005_2021.pkl'),
        'model_path': os.path.join(STOCKA_BASE_DIR, './ckpt/lightgbm_model_2015_2021.pkl'),
    })
    
    predictions = strategy.realtime_prediction(data_handler, last_trading_date, 20)

    if predictions is None:
        logger.warning("预测结果为空")
        return
    
    rank_list, code_list, name_list, score_list = [], [], [], []
    
    for prediction in predictions:
        pred_date = prediction['date']
        pred_result = prediction['predictions']
        
        if pred_date != last_trading_date:
            continue
        
        rank = 1
        for stock_code, pred_score in pred_result:
            stock_info = data_handler.get_stock_info(int(stock_code))
            if len(stock_info) == 0 or stock_info is None:
                stock_name = "股票名称获取失败"
            else:
                stock_name = stock_info['stock_name'].iloc[0]
            
            logger.info(f"rank={rank}, code={stock_code}, name={stock_name}, score={pred_score}")
            rank_list.append(rank)
            code_list.append(stock_code)
            name_list.append(stock_name)
            score_list.append(pred_score)
            
            rank += 1
            
    pred_df = pd.DataFrame({'rank': rank_list, 'code': code_list, 'name': name_list, 'score': score_list})
    pred_save_dir = os.path.join(STOCKA_BASE_DIR, 'signals')
    pred_save_path = os.path.join(pred_save_dir, f'{last_trading_date.strftime("%Y%m%d")}.csv')
    
    if not os.path.exists(pred_save_dir):
        os.makedirs(pred_save_dir)
    
    pred_df.to_csv(pred_save_path, index=False)
    logger.info(f"{last_trading_date} 的预测结果已保存到 {pred_save_path}")


if __name__ == '__main__':

    db_path = os.path.join(STOCKA_BASE_DIR, 'data/stock.db')

    data_handler = DataHandler(db_path)
    data_updater = DataUpdater(data_handler)
    factor_calculator = Alpha158()
    
    last_trading_date = data_updater.get_appropriate_end_date()
    # last_trading_date = date(2026, 2, 5)
    logger.info(f"最近交易日: {last_trading_date}")
    
    update_factors(data_handler, data_updater, last_trading_date, factor_calculator)
    
    get_top_n_stocks(data_handler, last_trading_date)
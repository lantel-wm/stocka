import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import date
from quant_framework import MLStrategy, LiveTrader
from quant_framework.utils.logger import get_logger
import pandas as pd

# # 设置环境变量，让所有模块的日志都输出到同一个文件
# os.environ['STOCKA_LOG_FILE'] = 'logs/realtime_example.log'

logger = get_logger(__name__)

STOCKA_BASE_DIR = "/home/zzy/projects/stocka"

# 加载股票名称映射
def load_stock_names():
    """加载股票代码到名称的映射"""
    stock_names = {}

    # 加载上海股票列表
    sh_file = os.path.join(STOCKA_BASE_DIR, "data/stock/list/sh_stock_list.csv")
    if os.path.exists(sh_file):
        df_sh = pd.read_csv(sh_file, encoding='utf-8')
        for _, row in df_sh.iterrows():
            code = str(row['证券代码'])
            name = row['证券简称']
            stock_names[code] = name

    # 加载深圳股票列表
    sz_file = os.path.join(STOCKA_BASE_DIR, "data/stock/list/sz_stock_list.csv")
    if os.path.exists(sz_file):
        df_sz = pd.read_csv(sz_file, encoding='utf-8')
        for _, row in df_sz.iterrows():
            code = str(row['A股代码'])
            name = row['A股简称']
            stock_names[code] = name

    logger.info(f"加载了 {len(stock_names)} 只股票的名称信息")
    return stock_names

# 加载股票名称
stock_name_map = load_stock_names()

# 1. 创建策略
strategy = MLStrategy(params={
    'model_path': os.path.join(STOCKA_BASE_DIR, 'examples/lightgbm_model.pkl'),
    'top_k': 20,
    'force_rebalance': True,
    # 'rebalance_days': 3,
    # 'stop_loss': 0.01,
    # 'stop_loss_check_daily': True,
})

strategy = MLStrategy({
    'model_path': os.path.join(STOCKA_BASE_DIR, './ckpt/lightgbm_model_2005_2021.pkl'),
    'weight_method': 'equal',
    # 'weight_method': 'score',
    'min_score': 0.1,
    'rebalance_days': 3,
    'top_k': 3,
    'stop_loss': 0.01,
})

# 2. 创建实盘交易调度器
trader = LiveTrader(
    strategy=strategy,
    data_dir=os.path.join(STOCKA_BASE_DIR, "data/stock/kline/day"),
    signal_output_dir="signals",
)

# 3. 运行（自动更新数据 + 生成信号，强制调仓）
result = trader.run_with_update(
    target_date=date.today(),
    force_rebalance=True,  # 强制调仓，忽略调仓周期
    export_format="csv"
)

# 4. 查看结果 - 只显示选中的股票代码和名称
run_result = result['run_result']
if run_result['signals_generated'] > 0:
    logger.info(f"\n{'='*60}")
    logger.info(f"选股结果（前 {strategy.params['top_k']} 名）")
    logger.info(f"{'='*60}")

    # 只显示买入信号的股票（这些就是选中的股票）
    buy_signals = [s for s in run_result['signals'] if s.signal_type == 'BUY']
    logger.info(f"{'排名':<5} {'股票代码':<10} {'股票名称':<12} {'预测分数'}")
    logger.info(f"{'-'*60}")

    for idx, signal in enumerate(buy_signals, 1):
        # 从reason中提取预测分数
        score_str = signal.reason.split('预测分数: ')[1] if '预测分数:' in signal.reason else 'N/A'
        # 获取股票名称
        stock_name = stock_name_map.get(signal.code, '未知')
        logger.info(f"{idx:<5} {signal.code:<10} {stock_name:<12} {score_str}")
else:
    logger.warning("未生成任何信号")
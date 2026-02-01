import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import date
from quant_framework import MLStrategy, LiveTrader

STOCKA_BASE_DIR = "/home/zzy/projects/stocka"

# 1. 创建策略
strategy = MLStrategy(params={
    'model_path': os.path.join(STOCKA_BASE_DIR, 'examples/lightgbm_model.pkl'),
    'top_k': 20,
    'rebalance_days': 3,
})

# 2. 创建实盘交易调度器
trader = LiveTrader(
    strategy=strategy,
    data_dir=os.path.join(STOCKA_BASE_DIR, "data/stock/kline/day"),
    signal_output_dir="signals",
)

# 3. 运行（自动更新数据 + 生成信号）
result = trader.run_with_update(
    target_date=date.today(),
    export_format="csv"
)

# 4. 查看结果
print(f"生成 {result['run_result']['signals_generated']} 个信号")
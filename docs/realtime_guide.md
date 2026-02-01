# 实盘交易功能使用指南

## 概述

realtime 模块提供了完整的实盘交易功能，包括：
- **在线数据更新**：从 akshare API 获取最新行情数据
- **增量更新**：智能检测现有数据，仅下载新增部分
- **策略执行**：调用策略生成交易信号
- **状态管理**：持久化运行状态，支持断点续传
- **信号导出**：支持导出到 CSV/JSON 文件

## 模块结构

```
quant_framework/realtime/
├── __init__.py          # 模块导出
├── data_updater.py      # 数据更新器
└── live_trader.py       # 实盘交易调度器
```

## 核心类

### 1. DataUpdater（数据更新器）

负责从在线数据源（akshare）获取最新行情数据并增量更新本地文件。

#### 主要方法

- `update_stock_data(stock_code, end_date)` - 更新单只股票数据
- `update_batch_stock_data(stock_codes, end_date)` - 批量更新股票数据
- `update_index_data(index_code, end_date)` - 更新指数数据

#### 使用示例

```python
from quant_framework import DataUpdater

# 创建数据更新器
updater = DataUpdater(
    output_dir="data/stock/kline/day",
    use_parquet=True,
    delay=0.5  # 请求间隔（秒）
)

# 更新单只股票
result = updater.update_stock_data("600000", end_date="20240102")
print(f"状态: {result['status']}")
print(f"消息: {result['message']}")

# 批量更新
results = updater.update_batch_stock_data(
    stock_codes=['600000', '000001', '000002'],
    end_date="20240102"
)
```

### 2. LiveTrader（实盘交易调度器）

协调整个实盘交易流程，包括数据更新、策略执行、信号生成和状态管理。

#### 主要方法

- `run(target_date, force_rebalance, export_format)` - 运行一次实盘流程
- `run_with_update(target_date, force_rebalance, export_format)` - 运行并自动更新数据
- `get_execution_summary()` - 获取执行摘要
- `print_execution_summary()` - 打印执行摘要
- `reset_state()` - 重置状态

#### 使用示例

```python
from datetime import date
from quant_framework import MLStrategy, LiveTrader

# 1. 创建策略
strategy_params = {
    'model_path': 'examples/lightgbm_model.pkl',
    'top_k': 10,
    'rebalance_days': 5,
    'weight_method': 'equal',
    'stock_pool': ['600000', '600004', '000001', '000002']
}
strategy = MLStrategy(params=strategy_params)

# 2. 创建实盘交易调度器
trader = LiveTrader(
    strategy=strategy,
    data_dir="data/stock/kline/day",
    signal_output_dir="signals",
    log_dir="logs",
    state_file="live_trader_state.json"
)

# 3. 运行（自动更新数据 + 生成信号）
result = trader.run_with_update(
    target_date=date.today(),
    force_rebalance=False,  # 不强制调仓，遵循调仓周期
    export_format="csv"     # 导出格式：csv, json, both
)

# 4. 查看结果
if result['run_result']['status'] == 'success':
    print(f"✓ 运行成功！生成 {result['run_result']['signals_generated']} 个信号")

# 5. 查看执行摘要
trader.print_execution_summary()
```

## 完整工作流程

### 步骤 1：准备策略

首先创建一个策略实例（MLStrategy 或其他策略）：

```python
from quant_framework import MLStrategy

strategy = MLStrategy(params={
    'model_path': 'models/lightgbm_model.pkl',
    'top_k': 10,           # 选股数量
    'rebalance_days': 5,   # 调仓周期（交易日）
    'weight_method': 'equal',
    'stop_loss': 0.05,     # 5% 止损
    'stop_loss_check_daily': True
})
```

### 步骤 2：创建 LiveTrader

```python
from quant_framework import LiveTrader

trader = LiveTrader(
    strategy=strategy,
    data_dir="data/stock/kline/day",      # 数据文件目录
    signal_output_dir="signals",          # 信号输出目录
    log_dir="logs",                       # 日志目录
    state_file="live_trader_state.json"   # 状态文件
)
```

### 步骤 3：运行实盘交易

```python
from datetime import date

# 运行并自动更新数据
result = trader.run_with_update(
    target_date=date.today(),
    force_rebalance=False,
    export_format="both"  # 同时导出 CSV 和 JSON
)
```

### 步骤 4：查看信号

信号将保存到 `signals/` 目录：

- `signals_20240102.csv` - CSV 格式信号文件
- `signals_20240102.json` - JSON 格式信号文件

CSV 格式示例：

```csv
date,code,signal_type,price,weight,reason
2024-01-02,600000,buy,10.50,0.10,模型预测分数: 0.8234
2024-01-02,000001,buy,15.20,0.10,模型预测分数: 0.7891
```

JSON 格式示例：

```json
{
  "date": "20240102",
  "total_signals": 2,
  "buy_signals": 2,
  "sell_signals": 0,
  "signals": [
    {
      "date": "2024-01-02",
      "code": "600000",
      "signal_type": "buy",
      "price": 10.50,
      "weight": 0.10,
      "reason": "模型预测分数: 0.8234"
    }
  ]
}
```

## 状态持久化

LiveTrader 会将运行状态保存到 JSON 文件，包括：

- `strategy_start_date` - 策略开始日期
- `last_trading_date` - 上次交易日期
- `trading_days_count` - 交易日计数
- `last_rebalance_date` - 上次调仓日期
- `execution_history` - 执行历史记录（最近 100 次）

状态文件示例：

```json
{
  "strategy_start_date": "2024-01-02",
  "last_trading_date": "2024-01-15",
  "trading_days_count": 10,
  "last_rebalance_date": "2024-01-12",
  "execution_history": [
    {
      "date": "2024-01-15",
      "is_rebalance": true,
      "signals": {
        "total": 5,
        "buy": 3,
        "sell": 2,
        "details": [...]
      }
    }
  ]
}
```

## 调仓周期

策略会根据 `rebalance_days` 参数自动判断是否为调仓日：

- **调仓日**：策略会执行完整的选股逻辑，生成交易信号
- **非调仓日**：策略仅检查止损，不进行调仓

可以通过 `force_rebalance=True` 强制调仓（忽略周期）。

## 执行摘要

使用 `print_execution_summary()` 查看执行统计：

```python
trader.print_execution_summary()
```

输出示例：

```
======================================================================
执行摘要
======================================================================
总执行次数: 20
调仓次数: 4
交易日计数: 20
上次调仓日: 2024-01-12

最近的执行记录:
  2024-01-15: [调仓], 5 个信号 (买入: 3, 卖出: 2)
  2024-01-14: [普通], 0 个信号 (买入: 0, 卖出: 0)
  2024-01-13: [普通], 0 个信号 (买入: 0, 卖出: 0)
======================================================================
```

## 重置状态

如需重新开始，可以重置状态：

```python
trader.reset_state()
```

这将清空所有历史记录，下次运行将作为首次运行。

## 依赖项

### 必需依赖

- `pandas` - 数据处理

### 可选依赖

- `akshare` - 在线数据源（用于数据更新）
- `lightgbm` - ML 策略（如果使用 MLStrategy）

安装命令：

```bash
pip install pandas
pip install akshare   # 可选
pip install lightgbm  # 可选
```

## 注意事项

1. **数据更新**：使用 akshare API 时请注意请求频率，避免被限制
2. **调仓判断**：调仓日的判断委托给策略，确保策略正确实现了 `is_rebalance_day()` 方法
3. **状态文件**：状态文件包含重要信息，请勿随意删除
4. **信号导出**：信号文件会覆盖同日期的旧文件，请及时备份

## 完整示例

参考 `examples/livetrader_example.py` 查看完整的使用示例。

## 验证安装

运行验证脚本确认模块正确安装：

```bash
python examples/verify_realtime.py
```

这将检查所有文件、类和方法是否正确创建。

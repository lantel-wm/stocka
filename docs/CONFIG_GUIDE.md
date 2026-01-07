# 配置文件说明

## 配置文件结构

配置文件 `config.yaml` 完全控制回测过程的所有参数。

### 1. 数据配置 (data)

```yaml
data:
  base_path: "data/stock/kline/day"     # 数据路径
  stock_list_path: "data/stock/list"   # 股票列表路径

  min_data_points: 100                  # 最少数据点数

  # 数据加载时间范围（应该包含回测范围）
  load_start_date: "2019-01-01"        # 数据开始日期
  load_end_date: "2023-12-31"          # 数据结束日期

  stock_whitelist: ['000001']          # 股票白名单（可选）
```

**重要说明**：
- `load_start_date` 和 `load_end_date` 定义了要加载的历史数据范围
- 这个范围应该比回测范围更宽，以确保策略有足够的历史数据来计算指标
- 例如：策略需要30日均线，那么 `load_start_date` 应该比回测开始日期早至少30天

### 2. 回测配置 (backtest)

```yaml
backtest:
  start_date: "2020-01-01"             # 回测开始日期
  end_date: "2023-12-31"               # 回测结束日期
  initial_capital: 1000000             # 初始资金
  benchmark: "000001"                  # 基准
  verbose: true                        # 是否显示详细输出
```

**时间范围关系**：
- 回测范围必须在数据加载范围内
- 回测开始日期 >= 数据加载开始日期
- 回测结束日期 <= 数据加载结束日期

### 3. 策略配置 (strategy)

```yaml
strategy:
  type: "SimpleMAStrategy"             # 策略类名

  params:
    window: 10                         # 策略参数
    max_position: 1
    min_bars: 10

  stock_list: ['000001']               # 策略股票列表（可选）
```

**可用策略类型**：
- `SimpleMAStrategy` - 简单均线策略
- `DoubleMAStrategy` - 双均线策略
- `MultiMAStrategy` - 多均线策略

### 4. 交易成本配置 (transaction_cost)

```yaml
transaction_cost:
  commission_rate: 0.0003              # 佣金率（万三）
  stamp_duty_rate: 0.001               # 印花税率（千分之一，仅卖出）
  min_commission: 5.0                  # 最低佣金
  slippage: 0.001                      # 滑点
```

### 5. 风险控制配置 (risk_control)

```yaml
risk_control:
  max_single_position_ratio: 1.0       # 单只股票最大仓位比例
  max_positions: 10                    # 最大持仓数量
  stop_loss_ratio: 0.05                # 止损比例
  take_profit_ratio: 0.15              # 止盈比例
```

### 6. 绩效分析配置 (performance)

```yaml
performance:
  risk_free_rate: 0.03                 # 无风险利率
  trading_days_per_year: 252           # 年化交易日数
```

### 7. 输出配置 (output)

```yaml
output:
  reports_path: "reports"              # 报告输出目录
  create_timestamp_dir: true           # 是否创建带时间戳的子目录

  export:
    trades: true                       # 导出交易记录CSV
    portfolio_history: true            # 导出持仓历史CSV
    detailed_positions: true           # 导出详细持仓CSV
    metrics: true                      # 导出绩效指标JSON
    config: true                       # 保存配置文件

  plots:
    equity_curve: true                 # 绘制资金曲线
    returns_distribution: true         # 绘制收益率分布
    drawdown: true                     # 绘制回撤图

  chart_format: "png"                  # 图表格式
```

## 使用示例

### 基本使用

```bash
# 使用默认配置文件
python backtest.py

# 使用指定配置文件
python backtest.py --config my_config.yaml
```

### 修改配置示例

#### 示例1：只测试平安银行，2020年回测

```yaml
data:
  stock_whitelist: ['000001']
  load_start_date: "2019-01-01"
  load_end_date: "2020-12-31"

backtest:
  start_date: "2020-01-01"
  end_date: "2020-12-31"

strategy:
  type: "SimpleMAStrategy"
  params:
    window: 10
    max_position: 1
    min_bars: 10
  stock_list: ['000001']
```

#### 示例2：多只股票，双均线策略

```yaml
data:
  stock_whitelist: ['000001', '000002', '600000', '600036', '600519']
  load_start_date: "2019-01-01"
  load_end_date: "2023-12-31"

backtest:
  start_date: "2020-01-01"
  end_date: "2023-12-31"

strategy:
  type: "DoubleMAStrategy"
  params:
    short_window: 10
    long_window: 30
    max_position: 3
    min_bars: 30
  stock_list: ['000001', '000002', '600000', '600036', '600519']
```

## 时间范围设置建议

### 确保有足够的历史数据

1. **查看策略的 `min_bars` 参数**
   - SimpleMAStrategy: `min_bars = window`
   - DoubleMAStrategy: `min_bars = long_window`
   - MultiMAStrategy: `min_bars = ma3`

2. **设置数据加载范围**
   - `load_start_date` = `backtest.start_date` - 至少 `min_bars` 天
   - 最好再预留一些余量，比如 `min_bars * 2`

3. **示例**
   ```yaml
   strategy:
     type: "DoubleMAStrategy"
     params:
       long_window: 30  # 需要30天历史数据

   data:
     load_start_date: "2019-01-01"  # 早于回测开始日期

   backtest:
     start_date: "2020-01-01"  # 2020-01-01 前30天 = 2019-12-02
                                # 2019-01-01 更早，数据充足
   ```

## 输出文件

运行回测后，报告目录包含：

```
reports/
└── backtest_YYYYMMDD_HHMMSS/
    ├── config.yaml              # 配置文件副本
    ├── trades.csv               # 交易记录
    ├── portfolio_history.csv    # 持仓历史
    ├── detailed_positions.csv   # 详细持仓
    ├── metrics.json             # 绩效指标
    ├── equity_curve.png         # 资金曲线图
    ├── returns_distribution.png # 收益率分布图
    └── drawdown.png             # 回撤图
```

## 配置文件验证

回测脚本会自动验证：
- 回测时间范围是否在数据加载范围内
- 如果不在范围内，会显示警告信息

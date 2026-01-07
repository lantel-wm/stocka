# API 文档

本文档提供核心模块的 API 参考。

## 目录

- [DataHandler](#datahandler)
- [BaseStrategy](#basestrategy)
- [Signal](#signal)
- [BacktestEngine](#backtestengine)
- [Portfolio](#portfolio)
- [TransactionCost](#transactioncost)
- [Config](#config)

## DataHandler

数据处理器类，负责加载和管理A股日线数据。

### 初始化

```python
DataHandler(
    data_path: str,
    min_data_points: int = 100,
    stock_whitelist: Optional[List[str]] = None
)
```

**参数：**
- `data_path`: CSV数据文件所在目录
- `min_data_points`: 最少数据点数，用于过滤股票
- `stock_whitelist`: 股票白名单（可选），只加载白名单中的股票

### 主要方法

#### load_data()

```python
load_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> None
```

加载CSV数据。

**参数：**
- `start_date`: 开始日期（YYYY-MM-DD）
- `end_date`: 结束日期（YYYY-MM-DD）

#### get_data_before()

```python
get_data_before(code: str, date: date) -> pd.DataFrame
```

获取指定日期之前的数据（避免未来函数）。

**参数：**
- `code`: 股票代码
- `date`: 截止日期

**返回：** 历史数据 DataFrame

#### get_daily_data()

```python
get_daily_data(date: date) -> pd.DataFrame
```

获取指定日期的所有股票数据。

**参数：**
- `date`: 交易日期

**返回：** 当日所有股票数据

#### get_all_codes()

```python
get_all_codes() -> List[str]
```

获取所有股票代码列表。

**返回：** 股票代码列表（字符串）

---

## BaseStrategy

策略基类，所有交易策略都应继承此类。

### 初始化

```python
BaseStrategy(params: Optional[Dict] = None)
```

**参数：**
- `params`: 策略参数字典

### 属性

- `name`: 策略名称（类名）
- `params`: 策略参数
- `min_bars`: 策略需要的最少bar数量
- `stock_list`: 策略股票列表（可选）

### 主要方法

#### on_bar()

```python
@abstractmethod
def on_bar(
    data_handler: DataHandler,
    current_date: date,
    portfolio: Portfolio
) -> List[Signal]
```

每日运行策略逻辑（抽象方法，必须由子类实现）。

**参数：**
- `data_handler`: 数据处理器对象
- `current_date`: 当前交易日期
- `portfolio`: 投资组合对象

**返回：** 交易信号列表

#### get_target_codes()

```python
def get_target_codes(data_handler: DataHandler) -> List[str]
```

获取策略应该使用的股票代码列表。

**参数：**
- `data_handler`: 数据处理器对象

**返回：** 股票代码列表

- 如果策略设置了 `stock_list`，返回策略的股票列表
- 否则返回 DataHandler 中的所有股票

#### calculate_indicators()

```python
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame
```

计算技术指标（可选方法，子类可以重写）。

**参数：**
- `df`: 包含OHLCV数据的DataFrame

**返回：** 添加了技术指标的DataFrame

#### reset()

```python
def reset() -> None
```

重置策略状态，用于多次回测或重新初始化。

---

## Signal

交易信号类，表示单个股票的买卖信号。

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `date` | date | 信号日期 |
| `code` | str | 股票代码 |
| `signal_type` | str | 信号类型：BUY/SELL/HOLD |
| `price` | float | 建议价格 |
| `weight` | float | 仓位权重（0-1） |
| `reason` | str | 信号原因 |

### 信号类型常量

```python
Signal.BUY   # 买入
Signal.SELL  # 卖出
Signal.HOLD  # 持有
```

### 方法

#### to_dict()

```python
def to_dict() -> dict
```

转换为字典格式。

---

## BacktestEngine

回测引擎，协调整个回测流程。

### 初始化

```python
BacktestEngine(
    data_handler: DataHandler,
    strategy: BaseStrategy,
    initial_capital: float = 1000000.0,
    max_single_position_ratio: float = 0.3,
    transaction_cost: Optional[TransactionCost] = None
)
```

**参数：**
- `data_handler`: 数据处理器
- `strategy`: 交易策略
- `initial_capital`: 初始资金
- `max_single_position_ratio`: 单只股票最大仓位比例
- `transaction_cost`: 交易成本模型（可选）

### 主要方法

#### run()

```python
run(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    verbose: bool = True
) -> Dict
```

运行回测。

**参数：**
- `start_date`: 开始日期（YYYY-MM-DD）
- `end_date`: 结束日期（YYYY-MM-DD）
- `verbose`: 是否打印进度信息

**返回：** 回测结果字典

#### get_results()

```python
get_results() -> Dict
```

获取回测结果。

#### get_equity_curve()

```python
get_equity_curve() -> pd.Series
```

获取资金曲线。

#### get_drawdown_series()

```python
get_drawdown_series() -> pd.Series
```

获取回撤序列。

#### reset()

```python
reset() -> None
```

重置回测引擎。

---

## Portfolio

投资组合类，管理资金、持仓和交易记录。

### 初始化

```python
Portfolio(
    initial_capital: float = 1000000.0,
    max_single_position_ratio: float = 0.3
)
```

**参数：**
- `initial_capital`: 初始资金
- `max_single_position_ratio`: 单只股票最大仓位比例

### 属性

- `initial_capital`: 初始资金
- `cash`: 当前现金余额
- `total_value`: 总资产
- `positions`: 持仓字典 {code: Position}
- `trades`: 交易历史列表

### 主要方法

#### get_position()

```python
get_position(code: str) -> Optional[Position]
```

获取指定股票的持仓。

#### get_cash()

```python
get_cash() -> float
```

获取当前现金余额。

#### get_all_positions()

```python
get_all_positions() -> Dict[str, Position]
```

获取所有持仓。

#### get_position_count()

```python
get_position_count() -> int
```

获取持仓数量。

#### get_daily_history()

```python
get_daily_history() -> List[dict]
```

获取每日历史记录。

#### get_trades()

```python
get_trades() -> List[Trade]
```

获取所有交易记录。

#### reset()

```python
reset() -> None
```

重置投资组合。

---

## TransactionCost

交易成本计算类。

### 初始化

```python
TransactionCost(
    commission_rate: float = 0.0003,
    stamp_duty_rate: float = 0.001,
    min_commission: float = 5.0,
    slippage: float = 0.001
)
```

**参数：**
- `commission_rate`: 佣金率（默认万三，即0.03%）
- `stamp_duty_rate`: 印花税率（仅卖出收取，默认千分之一）
- `min_commission`: 最低佣金（默认5元）
- `slippage`: 滑点（默认0.1%）

### 预定义成本模型

#### StandardCost

A股标准交易成本（万三佣金 + 千一印花税 + 0.1%滑点）。

```python
from quant_framework import StandardCost

cost = StandardCost()
```

#### LowCost

低交易成本（万一佣金 + 千一印花税 + 0.05%滑点）。

```python
from quant_framework import LowCost

cost = LowCost()
```

#### ZeroCost

零成本（用于测试）。

```python
from quant_framework import ZeroCost

cost = ZeroCost()
```

### 主要方法

#### calculate_buy_cost()

```python
calculate_buy_cost(amount: float) -> float
```

计算买入成本（佣金）。

#### calculate_sell_cost()

```python
calculate_sell_cost(amount: float) -> float
```

计算卖出成本（佣金 + 印花税）。

#### apply_slippage()

```python
apply_slippage(price: float, is_buy: bool) -> float
```

应用滑点。

**参数：**
- `price`: 原始价格
- `is_buy`: 是否为买入

**返回：** 应用滑点后的价格

---

## Config

配置类，管理回测框架的所有配置参数。

### 初始化

```python
Config(config_path: str = "config.yaml")
```

**参数：**
- `config_path`: 配置文件路径

### 主要方法

#### get()

```python
get(key: str, default: Any = None) -> Any
```

获取配置值（支持点号分隔的嵌套键）。

**示例：**
```python
# 获取回测开始日期
start_date = config.get('backtest.start_date')

# 获取策略参数
params = config.get('strategy.params', {})
```

#### set()

```python
set(key: str, value: Any) -> None
```

设置配置值。

**示例：**
```python
config.set('backtest.initial_capital', 2000000)
```

#### get_data_config()

```python
get_data_config() -> Dict[str, Any]
```

获取数据配置。

#### get_backtest_config()

```python
get_backtest_config() -> Dict[str, Any]
```

获取回测配置。

#### get_strategy_config()

```python
get_strategy_config() -> Dict[str, Any]
```

获取策略配置。

#### create_strategy()

```python
create_strategy() -> BaseStrategy
```

根据配置创建策略实例。

**返回：** 策略实例

**抛出：**
- `ValueError`: 如果策略类不存在或导入失败

#### save()

```python
save(save_path: Optional[str] = None) -> str
```

保存配置到文件。

#### reload()

```python
reload() -> None
```

重新加载配置文件。

#### to_dict()

```python
to_dict() -> Dict[str, Any]
```

返回配置的字典副本。

---

## 工具函数

### load_config()

```python
from quant_framework import load_config

config = load_config("my_config.yaml")
```

加载配置文件的便捷函数。

---

## 类型提示

框架使用 Python 类型提示，建议在 IDE 中启用类型检查。

```python
from typing import List, Dict, Optional
from datetime import date
from quant_framework import DataHandler, BaseStrategy, Signal, Portfolio

def on_bar(
    data_handler: DataHandler,
    current_date: date,
    portfolio: Portfolio
) -> List[Signal]:
    ...
```

---

## 相关文档

- [配置文件指南](CONFIG_GUIDE.md)
- [策略开发指南](STRATEGY_GUIDE.md)
- [README](../README.md)

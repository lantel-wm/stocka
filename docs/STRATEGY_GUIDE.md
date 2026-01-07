# 策略开发指南

本指南将帮助您开发自定义的交易策略。

## 目录

- [基础概念](#基础概念)
- [创建策略](#创建策略)
- [策略生命周期](#策略生命周期)
- [最佳实践](#最佳实践)
- [示例策略](#示例策略)
- [在配置文件中使用](#在配置文件中使用)

## 基础概念

### Signal（信号）

交易信号是策略与投资组合之间的桥梁。每个信号包含以下信息：

```python
from quant_framework import Signal

signal = Signal()
signal.date = date(2020, 1, 1)      # 信号日期
signal.code = "000001"              # 股票代码
signal.signal_type = Signal.BUY     # 信号类型：BUY/SELL/HOLD
signal.price = 15.50                 # 价格
signal.weight = 0.5                  # 仓位权重（0-1）
signal.reason = "金叉买入"           # 信号原因
```

### 信号类型

- `Signal.BUY` - 买入信号
- `Signal.SELL` - 卖出信号
- `Signal.HOLD` - 持有信号（通常不返回）

## 创建策略

### 基本结构

所有策略必须继承 `BaseStrategy` 并实现 `on_bar()` 方法：

```python
from quant_framework import BaseStrategy, Signal
from datetime import date
from typing import List

class MyStrategy(BaseStrategy):
    def on_bar(self, data_handler, current_date: date, portfolio) -> List[Signal]:
        """
        每日调用的策略逻辑

        Args:
            data_handler: 数据处理器对象
            current_date: 当前交易日期
            portfolio: 投资组合对象

        Returns:
            信号列表
        """
        signals = []

        # 1. 获取股票列表
        codes = self.get_target_codes(data_handler)

        # 2. 遍历每只股票
        for code in codes:
            # 3. 获取历史数据
            df = data_handler.get_data_before(code, current_date)

            # 4. 检查数据充足性
            if len(df) < self.min_bars:
                continue

            # 5. 计算指标
            # 你的逻辑...

            # 6. 生成信号
            if should_buy:
                signal = Signal()
                signal.date = current_date
                signal.code = code
                signal.signal_type = Signal.BUY
                signal.price = df['close'].iloc[-1]
                signal.weight = 1.0
                signal.reason = "买入原因"
                signals.append(signal)

        return signals
```

### 初始化参数

通过 `params` 参数传递策略配置：

```python
class MyStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        default_params = {
            'param1': 10,
            'param2': 0.5,
            'min_bars': 20  # 声明需要的历史数据量
        }

        if params:
            default_params.update(params)

        super().__init__(default_params)

        self.param1 = self.params['param1']
        self.param2 = self.params['param2']
```

## 策略生命周期

### 每个交易日调用顺序

```
1. on_bar() 被调用
   ↓
2. 策略生成信号列表
   ↓
3. Portfolio.update() 执行交易
   ↓
4. 更新持仓和资金
   ↓
5. 记录当日状态
```

### 数据访问规则

**⚠️ 重要：避免未来函数**

```python
# ✅ 正确：只使用当前日期之前的数据
df = data_handler.get_data_before(code, current_date)
current_price = df['close'].iloc[-1]

# ❌ 错误：使用未来数据
daily_data = data_handler.get_daily_data(current_date)
future_price = daily_data.loc[code, 'close']  # 这是当天的收盘价，不可用于生成信号
```

## 最佳实践

### 1. 使用 get_target_codes()

```python
# ✅ 推荐：使用策略的股票列表
codes = self.get_target_codes(data_handler)

# ❌ 不推荐：硬编码或直接获取所有代码
codes = data_handler.get_all_codes()
```

这样策略可以独立配置股票列表，与 DataHandler 的白名单解耦。

### 2. 设置合理的 min_bars

```python
def __init__(self, params: dict = None):
    default_params = {
        'window': 20,
        'min_bars': 20  # 至少需要20条数据来计算20日均线
    }
```

回测引擎会自动确保从开始日期往前有足够的历史数据。

### 3. 检查数据充足性

```python
def on_bar(self, data_handler, current_date, portfolio):
    for code in self.get_target_codes(data_handler):
        df = data_handler.get_data_before(code, current_date)

        # 检查数据是否充足
        if len(df) < self.min_bars:
            continue  # 跳过数据不足的股票

        # 继续处理...
```

### 4. 查询当前持仓

```python
def on_bar(self, data_handler, current_date, portfolio):
    # 获取当前持仓数量
    position_count = portfolio.get_position_count()

    # 获取特定股票的持仓
    position = portfolio.get_position(code)
    if position and position.shares > 0:
        # 已持有该股票
        current_shares = position.shares
    else:
        # 未持有该股票
        current_shares = 0
```

### 5. 控制仓位数量

```python
def on_bar(self, data_handler, current_date, portfolio):
    # 当前持仓数量
    current_count = portfolio.get_position_count()
    max_position = self.params.get('max_position', 3)

    for code in self.get_target_codes(data_handler):
        # 只在未达到最大持仓数时买入
        if should_buy and current_count < max_position:
            # 生成买入信号
            pass
```

## 示例策略

### 示例1：价格突破策略

```python
from quant_framework import BaseStrategy, Signal
from datetime import date
from typing import List

class BreakoutStrategy(BaseStrategy):
    """
    价格突破N日高点策略
    """

    def __init__(self, params: dict = None):
        default_params = {
            'lookback_period': 20,
            'breakout_ratio': 1.02,
            'max_position': 3,
            'min_bars': 20
        }

        if params:
            default_params.update(params)

        super().__init__(default_params)

        self.lookback_period = self.params['lookback_period']
        self.breakout_ratio = self.params['breakout_ratio']
        self.max_position = self.params['max_position']

    def on_bar(self, data_handler, current_date, portfolio):
        signals = []
        codes = self.get_target_codes(data_handler)

        # 当前持仓数量
        current_count = portfolio.get_position_count()

        for code in codes:
            df = data_handler.get_data_before(code, current_date)

            if len(df) < self.lookback_period + 1:
                continue

            # 获取过去N天的最高价
            lookback_data = df.tail(self.lookback_period)
            highest_price = lookback_data['high'].max()
            breakout_price = highest_price * self.breakout_ratio

            current_price = df['close'].iloc[-1]

            # 检查当前持仓
            position = portfolio.get_position(code)
            current_shares = position.shares if position else 0

            # 突破买入
            if current_price > breakout_price and current_shares == 0:
                if current_count < self.max_position:
                    signal = Signal()
                    signal.date = current_date
                    signal.code = code
                    signal.signal_type = Signal.BUY
                    signal.price = current_price
                    signal.weight = 1.0 / self.max_position
                    signal.reason = f"突破{self.lookback_period}日高点"
                    signals.append(signal)

            # 简单止损：跌破5日均线
            elif current_shares > 0 and len(df) >= 5:
                ma5 = df['close'].tail(5).mean()
                if current_price < ma5:
                    signal = Signal()
                    signal.date = current_date
                    signal.code = code
                    signal.signal_type = Signal.SELL
                    signal.price = current_price
                    signal.weight = 1.0
                    signal.reason = "跌破5日均线止损"
                    signals.append(signal)

        return signals
```

### 示例2：RSI动量策略

```python
from quant_framework import BaseStrategy, Signal
from quant_framework.strategy.indicators import Indicators
from datetime import date
from typing import List

class RSIStrategy(BaseStrategy):
    """
    RSI超买超卖策略
    """

    def __init__(self, params: dict = None):
        default_params = {
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'max_position': 3,
            'min_bars': 14
        }

        if params:
            default_params.update(params)

        super().__init__(default_params)

        self.rsi_period = self.params['rsi_period']
        self.oversold_threshold = self.params['oversold_threshold']
        self.overbought_threshold = self.params['overbought_threshold']
        self.max_position = self.params['max_position']

    def on_bar(self, data_handler, current_date, portfolio):
        signals = []
        codes = self.get_target_codes(data_handler)
        current_count = portfolio.get_position_count()

        for code in codes:
            df = data_handler.get_data_before(code, current_date)

            if len(df) < self.rsi_period + 1:
                continue

            # 计算RSI
            rsi = Indicators.rsi(df, period=self.rsi_period)

            if rsi.isna().all():
                continue

            current_rsi = rsi.iloc[-1]
            current_price = df['close'].iloc[-1]

            position = portfolio.get_position(code)
            current_shares = position.shares if position else 0

            # RSI超卖买入
            if current_rsi < self.oversold_threshold and current_shares == 0:
                if current_count < self.max_position:
                    signal = Signal()
                    signal.date = current_date
                    signal.code = code
                    signal.signal_type = Signal.BUY
                    signal.price = current_price
                    signal.weight = 1.0 / self.max_position
                    signal.reason = f"RSI超卖({current_rsi:.2f})"
                    signals.append(signal)

            # RSI超买卖出
            elif current_rsi > self.overbought_threshold and current_shares > 0:
                signal = Signal()
                signal.date = current_date
                signal.code = code
                signal.signal_type = Signal.SELL
                signal.price = current_price
                signal.weight = 1.0
                signal.reason = f"RSI超买({current_rsi:.2f})"
                signals.append(signal)

        return signals
```

## 在配置文件中使用

### 步骤1：将策略放在正确位置

将您的策略文件放在项目中，例如：
```
quant_framework/strategy/
├── __init__.py
├── base_strategy.py
├── ma_strategy.py
└── my_strategies.py  # 新文件
```

### 步骤2：在 __init__.py 中导入策略

编辑 `quant_framework/strategy/__init__.py`：

```python
from .my_strategies import BreakoutStrategy, RSIStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'SimpleMAStrategy',
    'DoubleMAStrategy',
    'MultiMAStrategy',
    'BreakoutStrategy',  # 添加
    'RSIStrategy'         # 添加
]
```

### 步骤3：更新配置管理模块

编辑 `quant_framework/utils/config.py`，修改 `STRATEGY_MODULE_PATH` 或添加多个路径：

```python
# 选项1：将所有策略放在一个模块
STRATEGY_MODULE_PATH = "quant_framework.strategy.my_strategies"

# 选项2：支持多个模块（需要修改代码）
STRATEGY_MODULES = [
    "quant_framework.strategy.ma_strategy",
    "quant_framework.strategy.my_strategies"
]
```

### 步骤4：在配置文件中使用

```yaml
strategy:
  type: "BreakoutStrategy"  # 使用类名
  params:
    lookback_period: 20
    breakout_ratio: 1.02
    max_position: 3
    min_bars: 20

  stock_list: ['000001', '000002', '600000']
```

### 步骤5：运行回测

```bash
python backtest.py --config config.yaml
```

## 调试技巧

### 打印调试信息

```python
def on_bar(self, data_handler, current_date, portfolio):
    if current_date.day == 1:  # 每月第一天打印
        print(f"Date: {current_date}")
        print(f"Portfolio value: {portfolio.total_value}")
        print(f"Position count: {portfolio.get_position_count()}")
```

### 记录信号原因

确保每个信号都有清晰的 `reason`，这样在查看交易记录时可以理解为什么执行了这笔交易。

```python
signal.reason = f"MA{self.short_window}上穿MA{self.long_window}"
```

### 检查信号数量

```python
def on_bar(self, data_handler, current_date, portfolio):
    signals = []

    # ... 生成信号 ...

    if len(signals) > 0:
        print(f"{current_date}: 生成了 {len(signals)} 个信号")

    return signals
```

## 常见问题

### Q: 策略没有产生任何交易？

A: 检查以下几点：
1. 数据是否充足（`len(df) >= min_bars`）
2. 股票列表是否正确（`get_target_codes()`）
3. 买入条件是否过于严格
4. 是否已经达到最大持仓数

### Q: 回测开始日期被调整了？

A: 这是因为 `min_bars` 设置导致的。确保 `data.load_start_date` 比 `backtest.start_date` 早至少 `min_bars` 天。

### Q: 如何实现分批买入？

A: 使用 `signal.weight` 控制每次买入的仓位比例：

```python
# 第一次买入50%
signal.weight = 0.5

# 第二次买入剩余50%
signal.weight = 0.5
```

## 总结

开发策略的关键要点：

1. ✅ 继承 `BaseStrategy`
2. ✅ 实现 `on_bar()` 方法
3. ✅ 使用 `get_target_codes()` 获取股票列表
4. ✅ 设置合理的 `min_bars`
5. ✅ 避免未来函数
6. ✅ 为每个信号添加清晰的 `reason`
7. ✅ 控制持仓数量和仓位大小

祝您策略开发顺利！

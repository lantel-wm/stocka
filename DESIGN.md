# A股日频量化回测与实盘框架设计文档

## 1. 项目概述

### 1.1 目标
构建一个简洁、准确且可扩展的A股日频量化回测框架，支持基于历史日线数据的策略回测，并具备实盘信号生成能力。

### 1.2 核心特性
- **数据驱动**：支持CSV格式的A股历史日线数据
- **日频回测**：基于日线级别的回测系统
- **实盘支持**：每日生成具体操作信号（无需实际交易接口）
- **简洁准确**：保持代码简洁的同时确保回测准确性
- **可扩展性**：易于添加新策略和指标

## 2. 系统架构设计

### 2.1 整体架构
```
┌─────────────────┐
│   数据管理层     │
│  (DataHandler)  │
└────────┬────────┘
         │
┌────────▼────────┐
│   策略层        │
│  (Strategy)     │
└────────┬────────┘
         │
┌────────▼────────┐    ┌─────────────────┐
│   投资组合层    │◄──►│   交易执行层    │
│  (Portfolio)    │    │  (Execution)    │
└────────┬────────┘    └─────────────────┘
         │
┌────────▼────────┐
│   回测引擎      │
│  (Backtest)     │
└────────┬────────┘
         │
┌────────▼────────┐
│   绩效分析层    │
│  (Performance)  │
└─────────────────┘
```

### 2.2 核心模块说明
1. **数据管理层(DataHandler)**：读取、清洗、管理日线数据
2. **策略层(Strategy)**：定义交易逻辑和信号生成规则
3. **投资组合层(Portfolio)**：管理仓位、资金和风险控制
4. **交易执行层(Execution)**：模拟交易执行，考虑交易成本
5. **回测引擎(Backtest)**：协调各模块运行回测
6. **绩效分析层(Performance)**：计算回测指标和生成报告

## 3. 详细设计

### 3.1 数据结构设计

#### 3.1.1 日线数据格式
```python
# 必需的CSV列名（中文列名）
required_columns = [
    '日期',      # 日期，格式：YYYY-MM-DD
    '股票代码',  # 股票代码，如：000001
    '开盘',      # 开盘价
    '收盘',      # 收盘价
    '最高',      # 最高价
    '最低',      # 最低价
    '成交量',    # 成交量（股）
    '成交额',    # 成交额（元）
]

# 可选的CSV列名（增强分析能力）
optional_columns = [
    '振幅',      # 振幅百分比
    '涨跌幅',    # 涨跌幅百分比
    '涨跌额',    # 涨跌额
    '换手率',    # 换手率百分比
]
```

#### 3.1.2 内部数据结构
```python
class BarData:
    """单日行情数据"""
    def __init__(self):
        self.date: datetime.date = None
        self.code: str = ""
        self.open: float = 0.0
        self.high: float = 0.0
        self.low: float = 0.0
        self.close: float = 0.0
        self.volume: float = 0.0
        self.amount: float = 0.0
        self.pre_close: float = 0.0  # 前收盘价
```

#### 3.1.3 交易信号结构
```python
class Signal:
    """交易信号"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    
    def __init__(self):
        self.date: datetime.date = None
        self.code: str = ""
        self.signal_type: str = ""  # buy/sell/hold
        self.price: float = 0.0     # 建议价格
        self.weight: float = 1.0    # 仓位权重（0-1）
        self.reason: str = ""       # 信号原因
```

### 3.2 核心类设计

#### 3.2.1 DataHandler类
```python
class DataHandler:
    """数据处理类"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = {}  # 存储所有股票数据
        
    def load_data(self, start_date=None, end_date=None):
        """加载CSV数据"""
        pass
    
    def get_stock_data(self, code: str) -> pd.DataFrame:
        """获取单只股票数据"""
        pass
    
    def get_multi_data(self, codes: list, date: datetime.date) -> dict:
        """获取多只股票在指定日期的数据"""
        pass
    
    def update_data(self, new_data: pd.DataFrame):
        """更新数据（用于实盘）"""
        pass
    
    def get_available_dates(self) -> list:
        """获取所有可用日期"""
        pass
```

#### 3.2.2 Strategy基类
```python
class BaseStrategy:
    """策略基类"""
    
    def __init__(self, params: dict = None):
        self.params = params or {}
        
    def on_bar(self, data_handler: DataHandler, date: datetime.date, 
               portfolio: Portfolio) -> List[Signal]:
        """
        每日运行策略逻辑
        返回交易信号列表
        """
        raise NotImplementedError
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        pass
```

#### 3.2.3 Portfolio类
```python
class Portfolio:
    """投资组合管理"""
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # 持仓 {code: shares}
        self.cash = initial_capital
        self.history = []    # 历史记录
        
    def update(self, date: datetime.date, signals: List[Signal], 
               data_handler: DataHandler):
        """根据信号更新投资组合"""
        pass
    
    def calculate_value(self, data_handler: DataHandler, 
                       date: datetime.date) -> float:
        """计算总资产价值"""
        pass
    
    def get_position_value(self, code: str, data_handler: DataHandler,
                          date: datetime.date) -> float:
        """计算单个持仓价值"""
        pass
```

#### 3.2.4 BacktestEngine类
```python
class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, data_handler: DataHandler, strategy: BaseStrategy,
                 initial_capital: float = 1000000.0):
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital)
        self.results = {}
        
    def run(self, start_date=None, end_date=None):
        """运行回测"""
        pass
    
    def get_results(self) -> dict:
        """获取回测结果"""
        pass
```

#### 3.2.5 Performance类
```python
class Performance:
    """绩效分析"""
    
    @staticmethod
    def calculate_metrics(portfolio_history: list, risk_free_rate: float = 0.02) -> dict:
        """
        计算绩效指标
        - 年化收益率
        - 年化波动率
        - 夏普比率
        - 最大回撤
        - 胜率
        - 盈亏比
        """
        pass
    
    @staticmethod
    def plot_equity_curve(portfolio_history: list):
        """绘制资金曲线"""
        pass
    
    @staticmethod
    def generate_report(metrics: dict, portfolio: Portfolio) -> str:
        """生成文本报告"""
        pass
```

#### 3.2.6 RealTimeSignalGenerator类
```python
class RealTimeSignalGenerator:
    """实盘信号生成器"""
    
    def __init__(self, data_handler: DataHandler, strategy: BaseStrategy):
        self.data_handler = data_handler
        self.strategy = strategy
        
    def generate_signals(self, date: datetime.date = None) -> List[Signal]:
        """生成当日交易信号"""
        pass
    
    def export_signals(self, signals: List[Signal], 
                       output_format: str = "csv") -> str:
        """导出信号到文件"""
        pass
```

### 3.3 数据预处理与清洗

```python
class DataProcessor:
    """数据预处理与清洗"""

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        - 去除缺失值
        - 处理异常值
        - 确保数据完整性
        """
        # 去除关键列的缺失值
        df = df.dropna(subset=['date', 'code', 'close'])

        # 处理异常值（例如价格为0或负数）
        for col in ['open', 'high', 'low', 'close']:
            df = df[df[col] > 0]

        # 确保最高价 >= 最低价
        df = df[df['high'] >= df['low']]

        # 确保收盘价在最高价和最低价之间
        df = df[(df['close'] <= df['high']) & (df['close'] >= df['low'])]

        return df

    @staticmethod
    def calculate_pre_close(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算前收盘价（用于计算涨跌幅）
        """
        df = df.sort_values('date')
        df['pre_close'] = df['close'].shift(1)
        return df
```

### 3.4 交易成本模型

```python
class TransactionCost:
    """交易成本计算（A股标准）"""

    def __init__(self,
                 commission_rate: float = 0.0003,  # 佣金率，默认万三（双向收取）
                 stamp_duty_rate: float = 0.001,   # 印花税率，默认千一（仅卖出收取）
                 min_commission: float = 5.0,      # 最低佣金，默认5元
                 slippage: float = 0.001):         # 滑点，默认0.1%
        """
        初始化交易成本参数

        A股交易成本说明：
        - 佣金：双向收取，最低5元，费率一般在万一到万三之间
        - 印花税：仅卖出收取，费率为千分之一
        - 过户费：双向收取，费率为万分之一（已包含在佣金中）
        """
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_commission = min_commission
        self.slippage = slippage

    def calculate_buy_cost(self, amount: float) -> float:
        """
        计算买入成本
        买入时：佣金（最低5元）
        """
        commission = max(amount * self.commission_rate, self.min_commission)
        return commission

    def calculate_sell_cost(self, amount: float) -> float:
        """
        计算卖出成本
        卖出时：佣金（最低5元）+ 印花税（千分之一）
        """
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty_rate
        return commission + stamp_duty

    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        应用滑点
        买入时价格向上滑点，卖出时价格向下滑点
        """
        if is_buy:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def calculate_total_cost(self, price: float, shares: int, is_buy: bool) -> float:
        """
        计算总交易成本
        :param price: 交易价格
        :param shares: 交易股数（A股以100股为一手，必须是100的整数倍）
        :param is_buy: 是否为买入
        :return: 总成本（包括滑点损失）
        """
        amount = price * shares

        # 应用滑点
        actual_price = self.apply_slippage(price, is_buy)
        actual_amount = actual_price * shares

        # 计算交易费用
        if is_buy:
            cost = self.calculate_buy_cost(actual_amount)
        else:
            cost = self.calculate_sell_cost(actual_amount)

        # 返回总成本（交易费用 + 滑点损失）
        slippage_cost = abs(actual_amount - amount)
        return cost + slippage_cost
```

### 3.4 示例策略实现

```python
class SimpleMAStrategy(BaseStrategy):
    """简单均线策略"""
    
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.short_window = self.params.get('short_window', 10)
        self.long_window = self.params.get('long_window', 30)
        self.holding_period = self.params.get('holding_period', 5)
        
    def on_bar(self, data_handler: DataHandler, date: datetime.date, 
               portfolio: Portfolio) -> List[Signal]:
        signals = []
        
        # 获取所有股票代码
        codes = data_handler.get_all_codes()
        
        for code in codes:
            # 获取历史数据
            df = data_handler.get_stock_data(code)
            if len(df) < self.long_window:
                continue
                
            # 计算均线
            df['MA_short'] = df['close'].rolling(self.short_window).mean()
            df['MA_long'] = df['close'].rolling(self.long_window).mean()
            
            current_price = df.iloc[-1]['close']
            prev_ma_short = df.iloc[-2]['MA_short']
            prev_ma_long = df.iloc[-2]['MA_long']
            current_ma_short = df.iloc[-1]['MA_short']
            current_ma_long = df.iloc[-1]['MA_long']
            
            # 生成信号
            signal = Signal()
            signal.date = date
            signal.code = code
            signal.price = current_price
            
            # 金叉买入
            if (prev_ma_short <= prev_ma_long and 
                current_ma_short > current_ma_long):
                signal.signal_type = Signal.BUY
                signal.reason = f"MA金叉: {self.short_window}/{self.long_window}"
                signals.append(signal)
            
            # 死叉卖出
            elif (prev_ma_short >= prev_ma_long and 
                  current_ma_short < current_ma_long):
                signal.signal_type = Signal.SELL
                signal.reason = f"MA死叉: {self.short_window}/{self.long_window}"
                signals.append(signal)
        
        return signals
```

## 4. 实现细节

### 4.1 文件结构
```
quant_framework/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── data_handler.py      # 数据管理
├── strategy/
│   ├── __init__.py
│   ├── base_strategy.py     # 策略基类
│   └── ma_strategy.py       # 示例策略
├── portfolio/
│   ├── __init__.py
│   └── portfolio.py         # 投资组合管理
├── execution/
│   ├── __init__.py
│   └── transaction_cost.py  # 交易成本
├── backtest/
│   ├── __init__.py
│   └── engine.py           # 回测引擎
├── performance/
│   ├── __init__.py
│   └── analyzer.py         # 绩效分析
├── realtime/
│   ├── __init__.py
│   └── signal_generator.py # 实盘信号生成
└── utils/
    ├── __init__.py
    └── helpers.py          # 工具函数
```

### 4.2 关键实现要点

#### 4.2.1 数据加载优化
```python
class DataHandler:
    # 定义列名映射（中文 -> 英文字段名）
    COLUMN_MAP = {
        '日期': 'date',
        '股票代码': 'code',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'amount',
        '振幅': 'amplitude',
        '涨跌幅': 'pct_change',
        '涨跌额': 'change_amount',
        '换手率': 'turnover'
    }

    def load_data(self):
        """优化数据加载，使用pandas提高效率"""
        # 批量读取所有CSV文件
        all_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        data_frames = []

        for file in all_files:
            # 读取CSV文件
            df = pd.read_csv(file)

            # 重命名列名为英文字段名（便于后续处理）
            df.rename(columns=self.COLUMN_MAP, inplace=True)

            # 如果文件名不是股票代码，则从数据中获取
            if 'code' not in df.columns or df['code'].isna().all():
                code = os.path.basename(file).replace('.csv', '')
                df['code'] = code

            # 确保日期格式正确
            df['date'] = pd.to_datetime(df['date'])

            data_frames.append(df)

        # 合并数据
        self.all_data = pd.concat(data_frames, ignore_index=True)

        # 创建多级索引
        self.all_data.set_index(['code', 'date'], inplace=True)
        self.all_data.sort_index(inplace=True)
```

#### 4.2.2 回测循环优化
```python
class BacktestEngine:
    def run(self):
        """高效的回测循环"""
        dates = self.data_handler.get_available_dates()
        dates.sort()
        
        for i, current_date in enumerate(dates):
            # 每日数据
            daily_data = self.data_handler.get_daily_data(current_date)
            
            # 策略生成信号
            signals = self.strategy.on_bar(
                self.data_handler, current_date, self.portfolio
            )
            
            # 更新投资组合
            self.portfolio.update(current_date, signals, self.data_handler)
            
            # 记录每日状态
            self.record_daily_status(current_date)
```

#### 4.2.3 实盘信号生成
```python
class RealTimeSignalGenerator:
    def generate_daily_report(self, date=None):
        """生成每日操作报告"""
        if date is None:
            date = datetime.date.today()
        
        # 获取最新数据
        signals = self.generate_signals(date)
        
        # 生成报告
        report = {
            'date': date.strftime('%Y-%m-%d'),
            'signals': [],
            'summary': {
                'buy_count': 0,
                'sell_count': 0,
                'hold_count': 0
            }
        }
        
        for signal in signals:
            report['signals'].append({
                'code': signal.code,
                'action': signal.signal_type,
                'price': signal.price,
                'reason': signal.reason
            })
            
            if signal.signal_type == Signal.BUY:
                report['summary']['buy_count'] += 1
            elif signal.signal_type == Signal.SELL:
                report['summary']['sell_count'] += 1
        
        return report
```

## 5. 使用示例

### 5.1 快速开始 - 最简单的回测示例

```python
import pandas as pd
from datetime import datetime

# ==================== 第1步：初始化数据处理器 ====================
data_handler = DataHandler("data/stock/kline/day")  # CSV文件所在目录
data_handler.load_data()

# ==================== 第2步：创建策略 ====================
# 使用最简单的双均线策略
strategy = SimpleMAStrategy({
    'short_window': 10,   # 短期均线
    'long_window': 30,    # 长期均线
    'max_position': 3     # 最多同时持有3只股票
})

# ==================== 第3步：运行回测 ====================
engine = BacktestEngine(
    data_handler=data_handler,
    strategy=strategy,
    initial_capital=1000000,  # 初始资金100万
    transaction_cost=TransactionCost()  # 使用默认交易成本
)

# 运行回测
engine.run(
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# ==================== 第4步：查看结果 ====================
# 获取回测结果
results = engine.get_results()

# 计算绩效指标
metrics = Performance.calculate_metrics(
    results['portfolio_history'],
    risk_free_rate=0.03  # 无风险利率3%
)

# 打印报告
print("\n" + "="*50)
print("回测绩效报告")
print("="*50)
print(f"初始资金: {1000000:,.2f} 元")
print(f"最终权益: {results['final_value']:,.2f} 元")
print(f"总收益率: {metrics['total_return']*100:.2f}%")
print(f"年化收益率: {metrics['annual_return']*100:.2f}%")
print(f"年化波动率: {metrics['annual_volatility']*100:.2f}%")
print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
print(f"胜率: {metrics['win_rate']*100:.2f}%")
print("="*50)

# 绘制资金曲线（可选）
Performance.plot_equity_curve(results['portfolio_history'])
```

### 5.2 完整回测示例（带自定义参数）

```python
# ==================== 自定义交易成本 ====================
custom_cost = TransactionCost(
    commission_rate=0.0003,  # 万三佣金
    stamp_duty_rate=0.001,   # 千一印花税
    min_commission=5.0,      # 最低佣金5元
    slippage=0.001           # 0.1%滑点
)

# ==================== 初始化并运行回测 ====================
engine = BacktestEngine(
    data_handler=data_handler,
    strategy=strategy,
    initial_capital=1000000,
    transaction_cost=custom_cost
)

engine.run(
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# ==================== 生成详细报告 ====================
results = engine.get_results()
metrics = Performance.calculate_metrics(results['portfolio_history'])

# 生成并保存HTML报告
report = Performance.generate_html_report(
    metrics=metrics,
    portfolio=engine.portfolio,
    trades=results['trades'],
    output_path='backtest_report.html'
)

print(f"报告已保存到: {report}")
```

### 5.3 实盘信号生成示例

```python
# ==================== 初始化实盘信号生成器 ====================
generator = RealTimeSignalGenerator(data_handler, strategy)

# ==================== 生成今日交易信号 ====================
# 方式1：生成当日信号
signals = generator.generate_signals()

# 方式2：生成指定日期的信号（用于测试）
from datetime import date
signals = generator.generate_signals(date=date(2024, 1, 5))

# ==================== 查看信号详情 ====================
print("\n今日交易信号:")
print("="*60)
for signal in signals:
    print(f"股票代码: {signal.code}")
    print(f"操作: {signal.signal_type}")  # 买入/卖出
    print(f"价格: {signal.price:.2f}")
    print(f"仓位权重: {signal.weight*100:.1f}%")
    print(f"原因: {signal.reason}")
    print("-"*60)

# ==================== 导出信号到文件 ====================
# 导出为CSV格式
csv_file = generator.export_signals(
    signals,
    output_format='csv',
    output_path='signals/daily_signals_20240105.csv'
)
print(f"\n信号已导出到: {csv_file}")

# 导出为JSON格式
json_file = generator.export_signals(
    signals,
    output_format='json',
    output_path='signals/daily_signals_20240105.json'
)
print(f"信号已导出到: {json_file}")

# ==================== 生成每日操作报告 ====================
report = generator.generate_daily_report()

print("\n" + "="*60)
print(f"日期: {report['date']}")
print("="*60)
print(f"买入信号: {report['summary']['buy_count']} 个")
print(f"卖出信号: {report['summary']['sell_count']} 个")
print(f"持有信号: {report['summary']['hold_count']} 个")
print("="*60)
```

### 5.4 创建自定义策略示例

```python
class MyCustomStrategy(BaseStrategy):
    """自定义策略示例：突破策略"""

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback_period = self.params.get('lookback_period', 20)
        self.breakout_ratio = self.params.get('breakout_ratio', 1.02)

    def on_bar(self, data_handler: DataHandler, date: datetime.date,
               portfolio: Portfolio) -> List[Signal]:
        signals = []

        # 获取所有股票代码
        codes = data_handler.get_all_codes()

        for code in codes:
            # 获取历史数据
            df = data_handler.get_stock_data(code)
            if len(df) < self.lookback_period:
                continue

            # 只使用历史数据（避免未来函数）
            df = df[df['date'] < date]
            if len(df) < self.lookback_period:
                continue

            # 获取最近的数据
            recent_data = df.tail(self.lookback_period)
            current_data = df[df['date'] == date]

            if len(current_data) == 0:
                continue

            # 计算突破信号
            highest_price = recent_data['high'].max()
            current_price = current_data.iloc[0]['close']

            # 生成信号
            signal = Signal()
            signal.date = date
            signal.code = code
            signal.price = current_price

            # 突破买入
            if current_price > highest_price * self.breakout_ratio:
                signal.signal_type = Signal.BUY
                signal.weight = 0.3  # 建议仓位30%
                signal.reason = f"突破{self.lookback_period}日高点"
                signals.append(signal)

        return signals

# 使用自定义策略
strategy = MyCustomStrategy({
    'lookback_period': 20,
    'breakout_ratio': 1.02
})
```

## 6. 准确性与优化考虑

### 6.1 回测准确性保障

#### 6.1.1 避免未来函数
```python
class BacktestEngine:
    def run(self, start_date=None, end_date=None):
        """
        严格按照时间顺序运行回测
        确保不使用未来数据
        """
        dates = self.data_handler.get_available_dates(start_date, end_date)
        dates.sort()

        for current_date in dates:
            # 只使用当前日期及之前的数据
            historical_data = self.data_handler.get_data_before(current_date)

            # 策略生成信号（只基于历史数据）
            signals = self.strategy.on_bar(
                self.data_handler, current_date, self.portfolio
            )

            # 执行交易
            self.portfolio.execute_trades(current_date, signals, self.data_handler)

            # 更新持仓价值
            self.portfolio.update_value(current_date, self.data_handler)
```

#### 6.1.2 处理涨跌停和停牌
```python
class TradingConstraints:
    """交易限制处理"""

    @staticmethod
    def is_limit_up(price: float, pre_close: float) -> bool:
        """
        判断是否涨停
        A股主板涨停限制为10%，创业板和科创板为20%
        """
        limit_rate = 0.10  # 主板10%
        return price >= pre_close * (1 + limit_rate) * 0.9995  # 考虑浮点误差

    @staticmethod
    def is_limit_down(price: float, pre_close: float) -> bool:
        """判断是否跌停"""
        limit_rate = 0.10
        return price <= pre_close * (1 - limit_rate) * 1.0005

    @staticmethod
    def is_suspended(df: pd.DataFrame, date: datetime.date) -> bool:
        """
        判断是否停牌
        如果当天没有交易数据，则视为停牌
        """
        if date in df['date'].values:
            return False
        return True

    @staticmethod
    def can_trade(code: str, date: datetime.date,
                  data_handler: DataHandler, signal_type: str) -> bool:
        """
        判断是否可以交易
        考虑涨跌停和停牌情况
        """
        df = data_handler.get_stock_data(code)

        # 检查是否停牌
        if TradingConstraints.is_suspended(df, date):
            return False

        # 获取当日数据
        daily_data = df[df['date'] == date].iloc[0]
        pre_close = daily_data['pre_close']

        # 买入时检查涨停
        if signal_type == Signal.BUY:
            if TradingConstraints.is_limit_up(daily_data['close'], pre_close):
                return False

        # 卖出时检查跌停
        elif signal_type == Signal.SELL:
            if TradingConstraints.is_limit_down(daily_data['close'], pre_close):
                return False

        return True
```

#### 6.1.3 精确的成本计算
```python
# 已在3.4节中详细实现
# 包括：
# 1. 佣金（双向，最低5元）
# 2. 印花税（仅卖出，千分之一）
# 3. 滑点（买入向上，卖出向下）
```

#### 6.1.4 流动性考虑
```python
class LiquidityChecker:
    """流动性检查"""

    @staticmethod
    def check_liquability(df: pd.DataFrame, date: datetime.date,
                         trade_amount: float, max_ratio: float = 0.1) -> bool:
        """
        检查流动性
        确保交易金额不超过当日成交额的一定比例（默认10%）
        """
        daily_data = df[df['date'] == date].iloc[0]
        daily_amount = daily_data['amount']  # 当日成交额

        return trade_amount <= daily_amount * max_ratio

    @staticmethod
    def calculate_max_shares(df: pd.DataFrame, date: datetime.date,
                            cash: float, price: float,
                            max_ratio: float = 0.1) -> int:
        """
        计算最大可买入股数
        考虑：
        1. 资金限制
        2. 流动性限制
        3. A股交易单位（100股为一手）
        """
        daily_data = df[df['date'] == date].iloc[0]
        daily_amount = daily_data['amount']

        # 基于流动性的最大金额
        max_amount_by_liquidity = daily_amount * max_ratio

        # 基于资金的最大金额
        max_amount_by_cash = cash

        # 取较小值
        max_amount = min(max_amount_by_liquidity, max_amount_by_cash)

        # 计算股数（向下取整到100的倍数）
        max_shares = int(max_amount / price / 100) * 100

        return max_shares
```

### 6.2 性能优化
1. **向量化操作**：使用pandas/numpy进行批量计算
2. **数据预处理**：在回测前计算所有技术指标
3. **缓存机制**：缓存常用计算结果
4. **并行处理**：多股票策略的并行计算

### 6.3 风险管理
1. **仓位控制**：单只股票最大仓位限制
2. **止损止盈**：支持止损止盈规则
3. **波动率调整**：根据市场波动调整仓位

## 7. 扩展性设计

### 7.1 添加新策略
```python
class YourCustomStrategy(BaseStrategy):
    def __init__(self, params=None):
        super().__init__(params)
        # 初始化参数
        
    def on_bar(self, data_handler, date, portfolio):
        # 实现策略逻辑
        signals = []
        # ... 策略计算 ...
        return signals
```

### 7.2 添加新指标
```python
def calculate_custom_indicator(df, window=20):
    """自定义指标计算函数"""
    # 实现指标逻辑
    return indicator_values
```

### 7.3 配置系统
```python
class Config:
    """配置管理类"""
    def __init__(self, config_file="config.yaml"):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key, default=None):
        return self.config.get(key, default)
```

## 8. 注意事项

### 8.1 数据相关

1. **数据质量**
   - 确保CSV数据格式正确，列名为中文
   - 检查数据完整性，无缺失值
   - 验证数据逻辑一致性（如最高价 >= 最低价）

2. **复权处理**
   - **强烈建议使用后复权数据进行回测**
   - 前复权可能导致早期价格失真
   - 不复权数据会受到分红送股影响，导致价格跳空

3. **分红送转**
   - 框架会在数据加载时自动处理除权除息
   - 使用后复权数据可以避免大部分问题
   - 如需精确计算分红收益，需要单独处理

4. **数据更新**
   - 实盘使用时需要每日更新数据
   - 确保数据日期的连续性
   - 处理停牌股票的数据缺失

### 8.2 交易机制

1. **A股交易规则**
   - T+1交易制度：当天买入的股票只能在下一个交易日卖出
   - 交易单位：100股为一手，必须是100的整数倍
   - 交易时间：9:30-11:30, 13:00-15:00

2. **涨跌停限制**
   - 主板股票：±10%
   - 创业板/科创板：±20%
   - ST股票：±5%
   - 新股前5个交易日：无涨跌停限制

3. **交易成本**
   - 佣金：双向收取，最低5元（通常万三）
   - 印花税：仅卖出收取（千分之一）
   - 过户费：双向收取（万分之一，已包含在佣金中）

### 8.3 回测注意事项

1. **避免未来函数**
   - 严格按时间顺序访问数据
   - 不能使用未来的技术指标
   - 策略逻辑必须仅基于历史数据

2. **过拟合风险**
   - 避免过度优化参数
   - 使用样本外测试验证
   - 简单策略往往优于复杂策略

3. **交易成本影响**
   - 小资金回测时，5元最低佣金影响巨大
   - 短期策略必须考虑交易成本
   - 高频策略在A股难以盈利

4. **流动性风险**
   - 小盘股流动性差，大额交易难以成交
   - 建议限制单只股票交易金额不超过成交额10%
   - 避免在停牌、涨跌停时交易

### 8.4 时间处理

1. **时区统一**
   - 所有日期使用北京时间（Asia/Shanghai）
   - 使用datetime.date而非datetime.datetime
   - 避免时区转换错误

2. **交易日历**
   - A股交易日：周一至周五（节假日除外）
   - 需要考虑节假日和周末
   - 建议使用交易日历而非自然日

### 8.5 性能与内存

1. **大数据集处理**
   - 数据量大时考虑使用数据库（如SQLite）
   - 使用pandas的chunk功能分批读取
   - 及时释放不需要的数据

2. **回测优化**
   - 预先计算技术指标，避免重复计算
   - 使用向量化操作替代循环
   - 考虑使用缓存机制

3. **并行计算**
   - 多策略回测可并行执行
   - 多股票回测可考虑并行处理
   - 注意GIL限制，某些操作可能无法并行

## 9. 快速参考

### 9.1 常用配置参数

```python
# 数据路径
DATA_PATH = "data/stock/kline/day"  # CSV文件目录

# 回测参数
INITIAL_CAPITAL = 1000000  # 初始资金100万
START_DATE = "2020-01-01"  # 回测开始日期
END_DATE = "2023-12-31"    # 回测结束日期

# 交易成本（A股标准）
COMMISSION_RATE = 0.0003   # 万三佣金
STAMP_DUTY_RATE = 0.001    # 千一印花税（仅卖出）
MIN_COMMISSION = 5.0       # 最低佣金5元
SLIPPAGE = 0.001           # 0.1%滑点

# 策略参数示例（双均线）
SHORT_WINDOW = 10          # 短期均线
LONG_WINDOW = 30           # 长期均线
MAX_POSITION = 5           # 最大持仓数

# 风险控制
MAX_SINGLE_POSITION_RATIO = 0.3  # 单只股票最大仓位30%
STOP_LOSS_RATIO = 0.05           # 止损5%
TAKE_PROFIT_RATIO = 0.15         # 止盈15%
```

### 9.2 核心类快速查找

| 类名 | 功能 | 模块路径 |
|------|------|----------|
| `DataHandler` | 数据加载与管理 | `data/data_handler.py` |
| `BaseStrategy` | 策略基类 | `strategy/base_strategy.py` |
| `SimpleMAStrategy` | 双均线策略示例 | `strategy/ma_strategy.py` |
| `Portfolio` | 投资组合管理 | `portfolio/portfolio.py` |
| `TransactionCost` | 交易成本计算 | `execution/transaction_cost.py` |
| `BacktestEngine` | 回测引擎 | `backtest/engine.py` |
| `Performance` | 绩效分析 | `performance/analyzer.py` |
| `RealTimeSignalGenerator` | 实盘信号生成 | `realtime/signal_generator.py` |

### 9.3 数据格式速查

**CSV文件格式（必需列）：**
```
日期,股票代码,开盘,收盘,最高,最低,成交量,成交额
2015-01-05,000001,1382.52,1385.12,1407.7,1348.66,2860436,4565387776.0
```

**CSV文件格式（可选列）：**
```
日期,股票代码,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
```

**内部数据结构（英文字段名）：**
```python
{
    'date': datetime.date(2015, 1, 5),
    'code': '000001',
    'open': 1382.52,
    'close': 1385.12,
    'high': 1407.7,
    'low': 1348.66,
    'volume': 2860436,
    'amount': 4565387776.0,
    'pre_close': 1369.50  # 前收盘价（自动计算）
}
```

### 9.4 常见问题排查

**问题1：回测收益与实际不符**
- 检查是否使用后复权数据
- 确认交易成本设置是否正确
- 验证是否考虑了涨跌停和停牌

**问题2：策略信号过多或过少**
- 调整策略参数（如均线周期）
- 检查数据是否完整
- 考虑添加过滤条件

**问题3：内存占用过大**
- 使用数据采样或减少股票数量
- 分批处理数据
- 考虑使用数据库

**问题4：回测速度慢**
- 预先计算技术指标
- 使用向量化操作
- 减少日志输出

## 10. 后续改进方向

1. **多因子策略**：支持多因子模型
2. **风险模型**：集成风险模型和约束条件
3. **参数优化**：自动参数优化和验证
4. **机器学习**：集成机器学习模型
5. **Web界面**：开发可视化Web界面

---

## 附录：完整项目结构

```
stocka/                          # 项目根目录
├── README.md                     # 项目说明文档
├── DESIGN.md                     # 设计文档（本文件）
├── requirements.txt              # Python依赖
├── config.yaml                   # 配置文件
│
├── data/                         # 数据目录
│   └── stock/
│       └── kline/
│           └── day/             # 日线数据
│               ├── 000001.csv   # 平安银行
│               ├── 000002.csv   # 万科A
│               └── ...
│
├── quant_framework/             # 框架核心代码
│   ├── __init__.py
│   │
│   ├── data/                    # 数据管理层
│   │   ├── __init__.py
│   │   ├── data_handler.py      # 数据管理
│   │   └── data_processor.py    # 数据预处理
│   │
│   ├── strategy/                # 策略层
│   │   ├── __init__.py
│   │   ├── base_strategy.py     # 策略基类
│   │   ├── ma_strategy.py       # 双均线策略
│   │   └── indicators.py        # 技术指标库
│   │
│   ├── portfolio/               # 投资组合层
│   │   ├── __init__.py
│   │   ├── portfolio.py         # 投资组合管理
│   │   └── position.py          # 持仓管理
│   │
│   ├── execution/               # 交易执行层
│   │   ├── __init__.py
│   │   ├── transaction_cost.py  # 交易成本
│   │   └── trade_executor.py    # 交易执行
│   │
│   ├── backtest/                # 回测引擎
│   │   ├── __init__.py
│   │   ├── engine.py            # 回测引擎
│   │   └── recorder.py          # 回测记录
│   │
│   ├── performance/             # 绩效分析层
│   │   ├── __init__.py
│   │   ├── analyzer.py          # 绩效分析
│   │   └── reports.py           # 报告生成
│   │
│   ├── realtime/                # 实盘信号生成
│   │   ├── __init__.py
│   │   └── signal_generator.py  # 信号生成器
│   │
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       ├── helpers.py           # 辅助函数
│       └── constraints.py       # 交易限制
│
├── examples/                    # 示例代码
│   ├── simple_backtest.py       # 简单回测示例
│   ├── custom_strategy.py       # 自定义策略示例
│   └── realtime_signals.py      # 实盘信号示例
│
├── signals/                     # 信号输出目录
│   ├── daily_signals_20240105.csv
│   └── ...
│
└── reports/                     # 报告输出目录
    ├── backtest_report_20240105.html
    └── ...
```

---

**文档版本**: v1.0
**最后更新**: 2025-01-05
**作者**: Stocka Team

# Stocka - A股量化回测框架

<div align="center">

一个简洁、准确且可扩展的A股日频量化回测框架

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[快速开始](#-快速开始) • [配置文件](#-配置文件) • [文档](#-文档) • [示例](#-示例)

</div>

## ✨ 特性

- 🎯 **准确性优先** - 严格遵循A股交易规则（T+1、涨跌停、交易单位）
- 📊 **配置驱动** - 通过 YAML 配置文件完全控制回测过程
- 🔌 **易于扩展** - 清晰的模块设计，轻松添加自定义策略
- 📈 **完整报告** - 自动生成 CSV、JSON 和图表报告
- 🚀 **命令行工具** - 一行命令运行完整回测

## 📋 系统要求

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- pyyaml >= 6.0

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将A股日线数据 CSV 文件放入 `data/stock/kline/day/` 目录。

**CSV 格式**：
```
日期,股票代码,开盘,收盘,最高,最低,成交量,成交额
2020-01-02,000001,15.20,15.50,15.60,15.10,1000000,15500000.0
```

### 3. 使用配置文件运行回测

```bash
# 使用默认配置文件
python backtest.py

# 使用自定义配置文件
python backtest.py --config my_config.yaml
```

### 4. 查看报告

回测完成后，报告将保存在 `reports/backtest_YYYYMMDD_HHMMSS/` 目录：

```
reports/backtest_20240107_120000/
├── config.yaml              # 配置文件副本
├── trades.csv               # 交易记录
├── portfolio_history.csv    # 持仓历史
├── detailed_positions.csv   # 详细持仓
├── metrics.json             # 绩效指标
├── equity_curve.png         # 资金曲线图
├── returns_distribution.png # 收益率分布图
└── drawdown.png             # 回撤图
```

## 🔧 配置文件

通过 `config.yaml` 完全控制回测过程：

```yaml
# 数据配置
data:
  base_path: "data/stock/kline/day"
  load_start_date: "2019-01-01"    # 数据加载开始日期
  load_end_date: "2023-12-31"      # 数据加载结束日期
  stock_whitelist: ['000001']      # 股票白名单（可选）

# 回测配置
backtest:
  start_date: "2020-01-01"         # 回测开始日期
  end_date: "2020-04-30"           # 回测结束日期
  initial_capital: 1000000         # 初始资金
  verbose: true

# 策略配置
strategy:
  type: "SimpleMAStrategy"         # 策略类名
  params:
    window: 10                     # 策略参数
    max_position: 1
    min_bars: 10
  stock_list: ['000001']           # 策略股票列表（可选）

# 交易成本配置
transaction_cost:
  commission_rate: 0.0003          # 万三佣金
  stamp_duty_rate: 0.001           # 千一印花税
  min_commission: 5.0              # 最低佣金
  slippage: 0.001                  # 滑点

# 风险控制
risk_control:
  max_single_position_ratio: 1.0   # 单只股票最大仓位
```

**📖 详细配置说明**: [docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md)

## 📖 文档

- [配置文件完整指南](docs/CONFIG_GUIDE.md) - 详细的配置说明和示例
- [策略开发指南](docs/STRATEGY_GUIDE.md) - 如何开发自定义策略
- [API 文档](docs/API.md) - 核心模块 API 参考

## 💡 使用示例

### 基础回测

```python
from quant_framework import DataHandler, SimpleMAStrategy, BacktestEngine

# 加载数据
data_handler = DataHandler(
    data_path="data/stock/kline/day",
    stock_whitelist=['000001']
)
data_handler.load_data(start_date="2019-01-01", end_date="2020-12-31")

# 创建策略
strategy = SimpleMAStrategy({'window': 10, 'max_position': 1, 'min_bars': 10})

# 运行回测
engine = BacktestEngine(data_handler, strategy, initial_capital=1000000)
results = engine.run(start_date="2020-01-01", end_date="2020-04-30")

print(f"总收益率: {results['total_return']*100:.2f}%")
```

### 自定义策略

```python
from quant_framework import BaseStrategy, Signal
from datetime import date
from typing import List

class MyStrategy(BaseStrategy):
    def on_bar(self, data_handler, current_date: date, portfolio) -> List[Signal]:
        signals = []
        codes = self.get_target_codes(data_handler)  # 获取策略的股票列表

        for code in codes:
            df = data_handler.get_data_before(code, current_date)
            if len(df) < 20:
                continue

            # 你的策略逻辑
            # ...

            signal = Signal()
            signal.date = current_date
            signal.code = code
            signal.signal_type = Signal.BUY
            signal.price = df['close'].iloc[-1]
            signal.reason = "买入原因"
            signals.append(signal)

        return signals
```

## 📁 项目结构

```
stocka/
├── quant_framework/          # 框架核心代码
│   ├── data/                # 数据管理层
│   ├── strategy/            # 策略层
│   ├── portfolio/           # 投资组合管理
│   ├── execution/           # 交易执行
│   ├── backtest/            # 回测引擎
│   ├── performance/         # 绩效分析
│   ├── realtime/            # 实盘信号生成
│   └── utils/               # 工具函数（配置管理等）
├── examples/                # 示例代码
├── docs/                    # 文档
│   ├── CONFIG_GUIDE.md      # 配置文件指南
│   ├── STRATEGY_GUIDE.md    # 策略开发指南
│   └── API.md               # API 文档
├── data/                    # 数据目录
├── reports/                 # 报告输出
├── signals/                 # 信号输出
├── backtest.py              # 命令行回测工具
├── config.yaml              # 配置文件
└── requirements.txt         # 依赖包
```

## 🎯 核心模块

| 模块 | 说明 |
|------|------|
| **DataHandler** | 数据加载和管理，支持股票白名单 |
| **BaseStrategy** | 策略基类，提供 get_target_codes() 等方法 |
| **BacktestEngine** | 回测引擎，自动处理 min_bars 数据要求 |
| **Portfolio** | 投资组合管理，T+1 限制，仓位控制 |
| **Performance** | 绩效分析，计算各项指标 |
| **Config** | 配置管理，从 YAML 创建策略实例 |

## 📊 内置策略

- **SimpleMAStrategy** - 简单均线策略
- **DoubleMAStrategy** - 双均线金叉死叉策略
- **MultiMAStrategy** - 多均线多头排列策略

## ⚠️ 重要提示

### 数据质量
- **强烈建议使用后复权数据**
- 确保数据格式正确，无缺失值

### A股交易规则
- **T+1 制度**：当天买入只能在次日卖出
- **交易单位**：100 股为一手
- **涨跌停限制**：主板 ±10%

### 交易成本
- 5 元最低佣金对小资金影响巨大
- 短期策略必须考虑交易成本

### 时间范围设置
- **数据加载范围**应该比**回测范围**更宽
- 确保有足够的历史数据供策略计算指标
- 根据 `min_bars` 参数确定需要的历史数据量

## 🛠️ 开发指南

### 添加新策略

1. 继承 `BaseStrategy`
2. 实现 `on_bar()` 方法
3. 使用 `get_target_codes()` 获取股票列表

详见 [策略开发指南](docs/STRATEGY_GUIDE.md)

### 配置驱动的策略

策略类名可以直接在配置文件中指定：

```yaml
strategy:
  type: "MyStrategy"  # 使用类名
  params:
    param1: value1
```

## 📝 待办事项

- [ ] 增加更多内置策略
- [ ] 支持分钟级回测
- [ ] 添加参数优化功能
- [ ] 支持多品种组合回测

## 📄 许可证

MIT License

## 👥 贡献

欢迎提交 Issue 和 Pull Request！

---

**版本**: 1.0.0
**最后更新**: 2025-01-07

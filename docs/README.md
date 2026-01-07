# Stocka 文档

欢迎来到 Stocka A股量化回测框架的文档中心！

## 📚 文档目录

### 快速开始
- [主 README](../README.md) - 项目概览和快速开始

### 配置指南
- [配置文件完整指南](CONFIG_GUIDE.md) - 详细的配置说明和最佳实践
  - 配置文件结构
  - 数据加载 vs 回测时间范围
  - 策略配置
  - 交易成本和风险控制
  - 输出选项

### 开发指南
- [策略开发指南](STRATEGY_GUIDE.md) - 如何开发自定义策略
  - 基础概念
  - 创建策略
  - 策略生命周期
  - 最佳实践
  - 示例策略
  - 调试技巧

### API 参考
- [API 文档](API.md) - 核心模块 API 参考
  - DataHandler
  - BaseStrategy
  - Signal
  - BacktestEngine
  - Portfolio
  - TransactionCost
  - Config

## 🚀 快速链接

### 我想...

#### 运行第一个回测
```bash
# 1. 查看主 README
cd /path/to/stocka
cat README.md

# 2. 准备数据
# 将CSV文件放入 data/stock/kline/day/

# 3. 运行回测
python backtest.py
```

#### 配置回测参数
→ 查看 [配置文件指南](CONFIG_GUIDE.md)

#### 开发自定义策略
→ 查看 [策略开发指南](STRATEGY_GUIDE.md)

#### 了解 API
→ 查看 [API 文档](API.md)

#### 解决常见问题
→ 查看各文档中的"常见问题"部分

## 📖 文档阅读顺序

### 初学者
1. [主 README](../README.md) - 了解项目
2. [配置文件指南](CONFIG_GUIDE.md) - 学习如何配置
3. 运行 `python backtest.py` - 实践

### 策略开发者
1. [策略开发指南](STRATEGY_GUIDE.md) - 学习如何开发策略
2. [API 文档](API.md) - 查阅 API 参考
3. 示例代码 (`examples/`) - 学习实战

### 高级用户
1. 所有文档
2. 源代码 (`quant_framework/`) - 深入了解实现

## 🎯 核心概念

### 数据加载范围 vs 回测范围

**重要概念**：数据加载范围应该比回测范围更宽！

```yaml
data:
  load_start_date: "2019-01-01"  # 数据加载（更早）
  load_end_date: "2023-12-31"

backtest:
  start_date: "2020-01-01"       # 回测（更晚）
  end_date: "2020-04-30"
```

**为什么？**
- 策略需要历史数据来计算指标（如均线）
- 例如：30日均线需要从回测开始日期往前30天的数据

### 股票白名单 vs 策略股票列表

- **DataHandler 股票白名单**：控制加载哪些股票的数据
- **策略股票列表**：控制策略对哪些股票生成信号

```python
# DataHandler: 只加载5只股票
data_handler = DataHandler(stock_whitelist=['000001', '000002', ...])

# Strategy: 只交易其中的3只
strategy = MyStrategy({'stock_list': ['000001', '000002', '000003']})
```

## 💡 最佳实践

### 1. 使用配置文件
推荐使用配置文件而不是硬编码：

```bash
# ✅ 推荐
python backtest.py --config my_strategy.yaml

# ❌ 不推荐
python examples/simple_backtest.py  # 需要修改代码
```

### 2. 设置合理的 min_bars
根据策略指标需求设置：

```python
# 30日均线策略
'min_bars': 30

# RSI(14) 策略
'min_bars': 14
```

### 3. 使用后复权数据
**强烈建议**使用后复权数据进行回测，以避免价格失真。

### 4. 控制仓位风险
```yaml
risk_control:
  max_single_position_ratio: 0.3  # 单只股票不超过30%
```

## 🔗 相关资源

### 项目文件
- [config.yaml](../config.yaml) - 默认配置文件
- [backtest.py](../backtest.py) - 命令行回测工具
- [requirements.txt](../requirements.txt) - 依赖包列表

### 示例代码
- `examples/simple_backtest.py` - 简单回测示例
- `examples/custom_strategy.py` - 自定义策略示例
- `examples/realtime_signals.py` - 实盘信号生成示例

### 源代码
- `quant_framework/data/` - 数据管理层
- `quant_framework/strategy/` - 策略层
- `quant_framework/backtest/` - 回测引擎
- `quant_framework/portfolio/` - 投资组合管理
- `quant_framework/performance/` - 绩效分析

## 🆘 获取帮助

### 常见问题

**Q: 回测开始日期为什么被调整了？**
A: 因为 `min_bars` 设置导致的。确保 `data.load_start_date` 比 `backtest.start_date` 早至少 `min_bars` 天。详见 [配置指南](CONFIG_GUIDE.md)。

**Q: 策略没有产生任何交易？**
A: 检查数据是否充足、股票列表是否正确、买入条件是否过于严格。详见 [策略开发指南](STRATEGY_GUIDE.md)。

**Q: 如何添加自定义策略？**
A: 继承 `BaseStrategy` 并实现 `on_bar()` 方法。详见 [策略开发指南](STRATEGY_GUIDE.md)。

### 报告问题

如果您发现 bug 或有功能建议，请：
1. 查看现有文档是否有答案
2. 搜索项目 Issues
3. 提交新的 Issue

## 📝 文档更新

- **最后更新**: 2025-01-07
- **版本**: 1.0.0

---

**祝您使用愉快！** 📈

# 机器学习量化交易Pipeline

## 概述

`MLPipeline` 是一个端到端的机器学习量化交易流程工具，整合了模型训练、模型评估和回测三个核心阶段。

## 功能特性

### 1. 模型训练
- 自动准备训练集和验证集
- 支持时间序列数据分割
- 训练LightGBM模型
- 自动保存模型到文件
- 记录训练日志

### 2. 模型评估
- 在测试集上进行预测
- 计算因子有效性指标：
  - IC (Information Coefficient) - 预测值与真实值的Pearson相关系数
  - IC均度 (IC_std) - IC的标准差
  - IC_IR (IC Information Ratio) - IC均值/IC标准差
  - RankIC - Spearman等级相关系数
  - RankIC_IR - RankIC的信息比率
- 生成评估报告

### 3. 回测
- 使用训练好的模型创建MLStrategy
- 运行完整回测
- 计算交易统计（胜率、盈亏比等）
- 生成回测报告和图表

## 快速开始

### 安装

Pipeline已集成在quant_framework中，无需额外安装。

### 基础用法

```python
from quant_framework import MLPipeline

# 配置Pipeline
config = {
    # 数据配置
    'data_path': 'path/to/data',

    # 日期配置
    'train_start': '2020-01-01',
    'train_end': '2022-12-31',
    'valid_start': '2023-01-01',
    'valid_end': '2023-12-31',
    'test_start': '2024-01-01',
    'test_end': '2024-12-31',
    'backtest_start': '2024-01-01',
    'backtest_end': '2024-12-31',

    # 模型参数
    'model_params': {
        'loss': 'mse',
        'num_boost_round': 1000,
        'early_stopping_rounds': 50,
    },

    # 策略参数
    'strategy_params': {
        'top_k': 10,
        'rebalance_days': 7,
        'stop_loss': 0.03,
    },

    # 回测参数
    'backtest_params': {
        'initial_capital': 1000000,
    },
}

# 创建并运行Pipeline
pipeline = MLPipeline(config)
results = pipeline.run()  # 一键运行完整流程
```

### 分步运行

```python
pipeline = MLPipeline(config)

# 分步运行各个阶段
training_info = pipeline.run_training()      # 1. 训练
eval_metrics = pipeline.run_evaluation()     # 2. 评估
backtest_results = pipeline.run_backtest()   # 3. 回测
pipeline.save_results()                      # 4. 保存结果
```

## 配置说明

### 必需配置项

| 参数 | 说明 | 示例 |
|------|------|------|
| `data_path` | 数据路径 | `'../data/factor/day/alpha158'` |
| `train_start` | 训练集开始日期 | `'2020-01-01'` |
| `train_end` | 训练集结束日期 | `'2022-12-31'` |
| `valid_start` | 验证集开始日期 | `'2023-01-01'` |
| `valid_end` | 验证集结束日期 | `'2023-12-31'` |
| `test_start` | 测试集开始日期 | `'2024-01-01'` |
| `test_end` | 测试集结束日期 | `'2024-12-31'` |
| `backtest_start` | 回测开始日期 | `'2024-01-01'` |
| `backtest_end` | 回测结束日期 | `'2024-12-31'` |

### 可选配置项

#### 模型参数 (`model_params`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `loss` | `'mse'` | 损失函数：'mse' 或 'binary' |
| `num_boost_round` | `1000` | 最大迭代轮数 |
| `early_stopping_rounds` | `50` | 早停轮数 |

#### 策略参数 (`strategy_params`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_k` | `10` | 选股数量 |
| `rebalance_days` | `5` | 调仓周期（交易日） |
| `weight_method` | `'equal'` | 仓位分配：'equal' 或 'score' |
| `stop_loss` | `0.05` | 止损阈值（5% = 0.05） |
| `stop_loss_check_daily` | `True` | 是否每日检查止损 |

#### 回测参数 (`backtest_params`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_capital` | `1000000` | 初始资金 |
| `max_single_position_ratio` | `1.0` | 单只股票最大仓位比例 |

## 输出结果

Pipeline运行后会在输出目录生成以下文件：

```
pipeline_outputs/
└── pipeline_20260131_120000/
    ├── model.pkl                    # 训练好的模型
    ├── evaluation_report.json       # IC等评估指标
    ├── trades.csv                   # 交易记录
    ├── portfolio_history.csv        # 持仓历史
    ├── trade_analysis.csv           # 交易详情分析
    ├── metrics.json                 # 回测绩效指标
    └── equity_curve.png             # 资金曲线图
```

## 评估指标说明

### IC (Information Coefficient)
- 预测值与真实值的Pearson相关系数
- 衡量预测的准确性
- 通常认为 IC > 0.03 为有效因子

### RankIC
- Spearman等级相关系数
- 对异常值更稳健
- 衡量预测的排序能力

### IC_IR (IC Information Ratio)
- IC均值 / IC标准差
- 衡量因子稳定性的重要指标
- 通常认为 IC_IR > 0.5 为稳定因子

## 示例

完整示例请参考 `examples/pipeline_example.py`

## 注意事项

1. **数据时序性**：确保训练集、验证集、测试集、回测集的时间顺序正确
2. **数据一致性**：所有数据集应使用相同的因子列表
3. **计算资源**：大规模数据建议调整 `num_workers` 参数
4. **过拟合风险**：验证集用于早停，测试集用于评估，避免数据泄露

## 扩展功能

未来计划添加的功能：
- 超参数搜索（网格搜索、贝叶斯优化）
- 交叉验证（TimeSeriesCV）
- 多模型对比
- 特征重要性分析
- 在线学习支持

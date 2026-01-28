# 因子分析模块指南

本指南介绍如何使用因子分析模块进行因子定义、计算和评估。

## 目录

- [模块概览](#模块概览)
- [BaseFactor - 因子基类](#basefactor---因子基类)
- [FactorMetrics - 因子指标计算](#factormetrics---因子指标计算)
- [MultiFactorAnalysis - 多因子分析](#multifactoranalysis---多因子分析)
- [使用示例](#使用示例)

## 模块概览

因子模块提供以下功能：

1. **因子定义和计算** - `BaseFactor` 类
2. **因子有效性评估** - `FactorMetrics` 类（IC、ICIR、Rank IC等）
3. **多因子分析** - `MultiFactorAnalysis` 类（因子合成、正交化、相关性分析）

## BaseFactor - 因子基类

所有自定义因子都应该继承 `BaseFactor` 类。

### 基本结构

```python
from quant_framework import BaseFactor
import pandas as pd

class MyFactor(BaseFactor):
    """自定义因子"""

    def __init__(self, name: str = "MyFactor", params: dict = None):
        super().__init__(name, params)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            因子值Series
        """
        # 实现因子计算逻辑
        # 例如：
        factor_values = data['close'] / data['close'].rolling(20).mean() - 1
        return factor_values
```

### 内置方法

#### 1. 数据验证

```python
factor = MyFactor()
is_valid = factor.validate_data(data)  # 检查数据是否包含必需列
```

#### 2. 缺失值处理

```python
# 前向填充
cleaned = factor.handle_missing_values(factor_values, method='ffill')

# 删除缺失值
cleaned = factor.handle_missing_values(factor_values, method='drop')

# 均值填充
cleaned = factor.handle_missing_values(factor_values, method='mean')
```

#### 3. 标准化

```python
# Z-score标准化
standardized = factor.standardize(factor_values, method='zscore')

# Min-Max标准化到[0,1]
standardized = factor.standardize(factor_values, method='minmax')

# 排名标准化（百分位）
standardized = factor.standardize(factor_values, method='rank')
```

#### 4. 去极值

```python
# 去除上下5%的极值
winsorized = factor.winsorize(factor_values, lower=0.05, upper=0.95)
```

#### 5. 因子中性化

```python
# 行业中性
neutralized = factor.neutralize(
    factor_values,
    industry=industry_series
)

# 市值中性
neutralized = factor.neutralize(
    factor_values,
    market_cap=market_cap_series
)
```

#### 6. 完整预处理流程

```python
# 自动预处理：去极值 -> 标准化
preprocessed = factor.preprocess(
    factor_values,
    winsorize=True,
    standardize=True,
    handle_missing=True
)
```

## FactorMetrics - 因子指标计算

用于评估因子的有效性。

### IC（Information Coefficient）

衡量因子值与未来收益率的相关性。

```python
from quant_framework import FactorMetrics

# 计算单期IC
ic = FactorMetrics.calculate_ic(
    factor_values,  # 因子值Series
    returns,        # 未来收益率Series
    method='pearson'  # 或 'spearman'
)
```

### Rank IC

因子排名与收益率排名的相关系数。

```python
rank_ic = FactorMetrics.calculate_rank_ic(factor_values, returns)
```

### ICIR（Information Coefficient Information Ratio）

ICIR = IC均值 / IC标准差

```python
ic_series = pd.Series([...])  # IC时间序列

icir = FactorMetrics.calculate_icir(ic_series, annualization=True)
```

### IC时间序列

```python
# factor_df: 因子DataFrame（索引为日期，列为股票代码）
# return_df: 收益率DataFrame（索引为日期，列为股票代码）

ic_series = FactorMetrics.calculate_ic_series(
    factor_df,
    return_df,
    periods=1,        # 预测期数
    method='pearson'
)
```

### 生成完整报告

```python
report = FactorMetrics.generate_report(
    factor_df,
    return_df,
    factor_name='MA偏离度因子',
    periods=1
)

# 打印报告
FactorMetrics.print_report(report)
```

报告包含：
- IC均值、标准差、ICIR
- Rank IC均值、标准差、ICIR
- IC>0占比
- 最大IC、最小IC

### 因子单调性检验

```python
# 将因子分为5组，检验各组收益的单调性
monotonicity = FactorMetrics.calculate_monotonicity(
    factor_values,
    returns,
    n_groups=5
)

print(f"是否单调: {monotonicity['monotonic']}")
print(f"各组收益: {monotonicity['group_returns']}")
print(f"多空收益: {monotonicity['long_short_return']}")
```

## MultiFactorAnalysis - 多因子分析

用于处理多个因子的组合和分析。

### 初始化

```python
from quant_framework import MultiFactorAnalysis

# factors是字典：{因子名: 因子DataFrame}
factors = {
    'MA5偏离度': ma5_factor_df,
    'MA30偏离度': ma30_factor_df,
    'RSI': rsi_factor_df
}

mfa = MultiFactorAnalysis(factors)
```

### 因子合成

```python
# 等权平均
combined = mfa.combine_factors(method='equal_weight')

# 加权求和
weights = {'MA5偏离度': 0.6, 'MA30偏离度': 0.4}
combined = mfa.combine_factors(method='weighted_sum', weights=weights)

# IC加权
combined = mfa.combine_factors(method='ic_weighted')

# 排名加权
combined = mfa.combine_factors(method='rank_weighted')
```

### 因子正交化

剔除参考因子的影响，得到相互独立的因子。

```python
# 回归法
orthogonalized = mfa.orthogonalize_factors(
    reference_factor='MA5偏离度',
    method='regression'
)

# Gram-Schmidt正交化
orthogonalized = mfa.orthogonalize_factors(
    reference_factor='MA5偏离度',
    method='gram_schmidt'
)

# PCA正交化
orthogonalized = mfa.orthogonalize_factors(
    reference_factor='MA5偏离度',
    method='pca'
)
```

### 因子相关性分析

```python
# 计算因子之间的相关性矩阵
corr_matrix = mfa.calculate_factor_correlation(method='pearson')
print(corr_matrix)
```

### 选择最优因子

```python
# 根据IC均值选择前5个因子
best_factors = mfa.select_best_factors(
    return_df,
    top_n=5,
    metric='ic_mean'  # 可选: 'ic_mean', 'rank_ic_mean', 'ic_ir', 'rank_ic_ir'
)

print(f"最优因子: {best_factors}")
```

### 多因子综合报告

```python
# 生成包含所有因子指标的DataFrame
combined_report = mfa.generate_combined_report(return_df)

# 打印综合报告
MultiFactorAnalysis.print_combined_report(combined_report)
```

## 使用示例

### 示例1：创建自定义因子

```python
from quant_framework import BaseFactor
import pandas as pd

class MomentumFactor(BaseFactor):
    """动量因子：过去N天的收益率"""

    def __init__(self, params: dict = None):
        default_params = {'period': 20}
        if params:
            default_params.update(params)
        super().__init__("Momentum", default_params)
        self.period = self.params['period']

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        # 计算过去N天的收益率
        momentum = (data['close'] - data['close'].shift(self.period)) / data['close'].shift(self.period)
        return momentum

# 使用因子
factor = MomentumFactor({'period': 20})
factor_values = factor.calculate(data)
```

### 示例2：因子评估流程

```python
from quant_framework import DataHandler, FactorMetrics

# 1. 加载数据
data_handler = DataHandler("data/stock/kline/day")
data_handler.load_data(start_date="2019-01-01", end_date="2020-12-31")

# 2. 计算因子值（假设已经计算好）
# factor_df 和 return_df 都是 DataFrame，索引为日期，列为股票代码

# 3. 评估因子
report = FactorMetrics.generate_report(
    factor_df,
    return_df,
    factor_name='我的因子',
    periods=1
)

# 4. 打印报告
FactorMetrics.print_report(report)

# 5. 分析因子单调性
monotonicity = FactorMetrics.calculate_monotonicity(
    factor_df.iloc[-1],  # 最新一期因子值
    return_df.iloc[-1],   # 最新一期收益率
    n_groups=5
)
```

### 示例3：多因子选股

```python
from quant_framework import MultiFactorAnalysis

# 1. 准备多个因子
factors = {
    '动量因子': momentum_df,
    '价值因子': value_df,
    '质量因子': quality_df
}

# 2. 创建多因子分析对象
mfa = MultiFactorAnalysis(factors)

# 3. 合并因子
combined_factor = mfa.combine_factors(method='ic_weighted')

# 4. 选股（选择因子值最高的N只股票）
for date in combined_factor.index:
    top_stocks = combined_factor.loc[date].nlargest(10)
    print(f"{date}: {top_stocks.index.tolist()}")
```

### 示例4：因子正交化

```python
# 假设动量因子和价值因子相关性较高
# 我们希望剔除动量因子的影响，得到纯粹的价值因子

orthogonalized = mfa.orthogonalize_factors(
    reference_factor='动量因子',
    method='regression'
)

# 获取正交化后的价值因子
pure_value_factor = orthogonalized['价值因子']
```

## 因子开发建议

### 1. 因子计算

- 使用 `BaseFactor` 基类
- 实现 `calculate()` 方法
- 确保返回 pandas Series

### 2. 因子预处理

推荐流程：
```python
preprocessed = factor.preprocess(
    factor_values,
    winsorize=True,      # 去极值
    standardize=True,    # 标准化
    handle_missing=True  # 处理缺失值
)
```

### 3. 因子评估

- 关注 **IC均值**：绝对值越大越好（通常>0.03有意义）
- 关注 **ICIR**：衡量IC的稳定性，通常>0.5较好
- 关注 **IC>0占比**：衡量因子的方向正确率，通常>55%较好
- 检验因子单调性：确保因子分组收益具有单调性

### 4. 多因子组合

- 对于相关性较高的因子，进行正交化处理
- 使用 IC 加权合成因子
- 定期（如每月）重新评估因子有效性

## 运行示例

```bash
# 运行因子分析示例
cd examples
python factor_analysis_example.py
```

## 注意事项

1. **避免未来函数**：计算因子时只能使用当前及之前的数据
2. **数据对齐**：确保因子值和收益率的索引对齐
3. **缺失值处理**：合理处理缺失值，避免偏差
4. **过拟合风险**：避免过度优化因子参数

## 相关文档

- [策略开发指南](STRATEGY_GUIDE.md) - 如何将因子用于策略开发
- [API 文档](API.md) - 因子相关 API 参考
- [配置文件指南](CONFIG_GUIDE.md) - 配置文件使用

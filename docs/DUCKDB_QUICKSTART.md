# DuckDB 快速开始指南

## 5分钟快速上手DuckDB

### 步骤1: 安装依赖

```bash
pip install duckdb
```

### 步骤2: 迁移数据到DuckDB

#### 迁移股票行情数据

```bash
python scripts/migrate_to_duckdb.py \
    --source data/stock/day \
    --target data/stock.db \
    --table stock_prices \
    --workers 8
```

#### 迁移Alpha158因子数据(可选)

```bash
python scripts/migrate_to_duckdb.py \
    --source data/factor/day/alpha158 \
    --target data/factor.db \
    --table alpha158_factors \
    --type factor \
    --workers 8
```

### 步骤3: 使用DatabaseDataHandler

```python
from quant_framework.data.database_data_handler import DatabaseDataHandler
from datetime import date

# 创建数据处理器
handler = DatabaseDataHandler('data/stock.db')

# 查询单只股票
df = handler.get_stock_data('000001')

# 查询截面数据
df = handler.get_daily_data(date(2024, 1, 10))

# 使用SQL查询
sql = "SELECT * FROM stock_prices WHERE code = '000001' ORDER BY date DESC LIMIT 10"
result = handler.execute_sql(sql)
```

### 步骤4: 运行性能测试

```bash
python scripts/benchmark_duckdb.py
```

## 核心优势

| 特性 | Parquet | DuckDB | 提升 |
|------|---------|--------|------|
| 查询速度 | 0.5秒 | 0.05秒 | **10x** |
| 存储空间 | 6GB | 2-3GB | **节省50%** |
| SQL支持 | ❌ | ✅ | **新增** |
| 部署难度 | 中等 | 简单 | **更容易** |

## 代码迁移

只需修改两行代码:

```python
# 旧代码
from quant_framework.data.data_handler import DataHandler
handler = DataHandler('data/stock/day')
handler.load_data()

# 新代码
from quant_framework.data.database_data_handler import DatabaseDataHandler
handler = DatabaseDataHandler('data/stock.db')

# 其他代码保持不变!
df = handler.get_stock_data('000001')
```

## 常见操作

### 批量查询(高性能)

```python
# 批量获取多只股票的数据
df = handler.get_data_range(
    codes=['000001', '000002', '600000'],
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)
```

### 因子数据查询

```python
from quant_framework.data.database_data_handler import FactorDataHandler

factor_handler = FactorDataHandler('data/factor.db')

# 获取截面因子
cross_section = factor_handler.get_factor_cross_section(date(2024, 1, 10))

# 批量获取因子
factors = factor_handler.get_factors(
    codes=['000001', '000002'],
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)
```

### 复杂SQL查询

```python
# 查询涨幅TOP10股票
sql = """
SELECT code,
       (LAST(close) - FIRST(close)) / FIRST(close) as return_rate
FROM stock_prices
WHERE date >= '2024-01-01' AND date <= '2024-12-31'
GROUP BY code
ORDER BY return_rate DESC
LIMIT 10
"""
result = handler.execute_sql(sql)
```

## 运行示例

```bash
python examples/duckdb_example.py
```

## 数据备份

DuckDB数据库是单个文件,可以直接复制:

```bash
# 备份
cp data/stock.db data/stock_backup.db

# 恢复
cp data/stock_backup.db data/stock.db
```

## 因子数据管理

### 初始化因子表

在 stock.db 中创建因子表结构并注册 Alpha158 因子定义：

```bash
python scripts/init_factor_tables.py --db-path data/stock.db
```

### 迁移因子数据

将 Parquet 格式的因子数据迁移到数据库：

```bash
python scripts/migrate_parquet_factors.py \
    --source data/factor/day/alpha158 \
    --db data/stock.db \
    --workers 8
```

### 使用因子功能

```python
from quant_framework.data.data_handler import DataHandler
from datetime import date

handler = DataHandler('data/stock.db')

# 初始化因子表（首次使用）
handler._init_factor_tables()

# 注册因子定义
handler.register_factor('MA5', 'alpha158', '5日移动平均')

# 保存因子值
factor_df = pd.DataFrame({
    'MA5': [1.02, 0.98, 1.01],
    'MA10': [0.99, 1.02, 0.97]
}, index=['000001', '000002', '600000'])

handler.save_factors(factor_df, date(2024, 1, 10))

# 查询截面因子（某日所有股票的因子）
cross_section = handler.get_factor_cross_section(
    date(2024, 1, 10),
    factor_names=['MA5', 'MA10']
)

# 查询时序因子（某股票的历史因子）
stock_factors = handler.get_stock_factors(
    '000001',
    factor_names=['MA5', 'MA10'],
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31)
)

# 查询宽表格式（用于ML预测）
wide_df = handler.get_factors_wide(
    trade_date=date(2024, 1, 10),
    stock_codes=['000001', '000002'],
    factor_names=['MA5', 'MA10']
)

# 获取因子定义信息
factor_info = handler.get_factor_info()
print(factor_info)
```

## 下一步

- 阅读详细文档: [DuckDB迁移指南](DUCKDB_MIGRATION.md)
- 运行性能测试: `python scripts/benchmark_duckdb.py`
- 查看代码示例: `examples/duckdb_example.py`
- 运行单元测试: `pytest tests/test_factor_handler.py -v`

## 问题排查

### 数据库文件不存在

```bash
# 确保先运行迁移脚本
python scripts/migrate_to_duckdb.py --source data/stock/day --target data/stock.db
```

### 查询速度慢

- 确保使用了批量查询方法`get_data_range()`
- 避免循环查询单个股票
- 使用日期过滤减少数据量

### 内存不足

- 使用`stock_whitelist`参数限制股票数量
- 在查询时指定日期范围

## 总结

✅ **简单**: 无需部署,嵌入式数据库
✅ **快速**: 查询性能提升10倍
✅ **兼容**: 代码改动最小
✅ **强大**: 支持复杂SQL查询

立即开始使用,享受性能提升!

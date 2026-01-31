# DataHandler 使用指南

## 概述

`DataHandler` 是一个统一的数据处理器，支持：
- ✅ **CSV 和 Parquet 格式**（自动检测）
- ✅ **单进程和多进程并行加载**
- ✅ **股票白名单过滤**
- ✅ **日期范围过滤**

## 基本用法

### 1. 自动检测格式（推荐）

```python
from quant_framework.data.data_handler import DataHandler

# 自动检测 CSV 或 Parquet 格式
handler = DataHandler(data_path='data/factor/day/alpha158')
handler.load_data()
```

### 2. 强制使用 Parquet 格式

```python
# 强制使用 Parquet（如果目录中同时有 CSV 和 Parquet）
handler = DataHandler(
    data_path='data/factor/day/alpha158',
    use_parquet=True  # True=Parquet, False=CSV, None=自动检测
)
handler.load_data()
```

### 3. 多进程并行加载（最快）

```python
# 使用 8 个进程并行加载（推荐用于 Parquet）
handler = DataHandler(
    data_path='data/factor/day/alpha158_parquet',
    use_parquet=True,
    num_workers=8  # 8个进程并行
)
handler.load_data()
```

### 4. 股票白名单

```python
# 只加载指定的几只股票
handler = DataHandler(
    data_path='data/factor/day/alpha158',
    stock_whitelist=['000001', '000002', '600000'],
    num_workers=4
)
handler.load_data()
```

### 5. 日期范围过滤

```python
# 只加载最近一年的数据
handler = DataHandler(
    data_path='data/factor/day/alpha158_parquet',
    num_workers=8
)
handler.load_data(
    start_date='2023-01-01',
    end_date='2024-01-01'
)
```

## 参数说明

### `__init__` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_path` | str | 必填 | 数据文件所在目录 |
| `min_data_points` | int | 100 | 最少数据点数，用于过滤股票 |
| `stock_whitelist` | List[str] | None | 股票白名单，只加载这些股票 |
| `use_parquet` | bool | None | None=自动检测，True=Parquet，False=CSV |
| `num_workers` | int | 1 | 并行进程数（1=单进程，>1=多进程） |

### `load_data` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `start_date` | str | None | 开始日期 (YYYY-MM-DD) |
| `end_date` | str | None | 结束日期 (YYYY-MM-DD) |

## 性能对比

### CSV 模式（单进程）

```python
handler = DataHandler(
    data_path='data/factor/day/alpha158',
    use_parquet=False,
    num_workers=1
)
handler.load_data()
# 速度: ~150万行/秒
```

### Parquet 模式（单进程）

```python
handler = DataHandler(
    data_path='data/factor/day/alpha158_parquet',
    use_parquet=True,
    num_workers=1
)
handler.load_data()
# 速度: ~400万行/秒 (2.7x 提升)
```

### Parquet 模式（8进程）

```python
handler = DataHandler(
    data_path='data/factor/day/alpha158_parquet',
    use_parquet=True,
    num_workers=8
)
handler.load_data()
# 速度: ~1200万行/秒 (8x 提升)
```

## 推荐配置

### 开发环境（CPU核心较少）

```python
handler = DataHandler(
    data_path='data/factor/day/alpha158',
    num_workers=4  # 4个进程
)
```

### 生产环境（CPU核心较多）

```python
handler = DataHandler(
    data_path='data/factor/day/alpha158_parquet',
    use_parquet=True,
    num_workers=16  # 16个进程
)
```

### 小数据集或内存有限

```python
handler = DataHandler(
    data_path='data/factor/day/alpha158',
    num_workers=1,  # 单进程节省内存
    stock_whitelist=['000001', '600000']  # 只加载需要的股票
)
```

## 迁移指南

### 从旧版 DataHandler 迁移

**旧代码：**
```python
from quant_framework.data.data_handler import DataHandler

handler = DataHandler(data_path='data/factor/day/alpha158')
handler.load_data()
```

**新代码（完全兼容）：**
```python
from quant_framework.data.data_handler import DataHandler

# 代码完全相同，但内部会自动检测 Parquet 格式
handler = DataHandler(data_path='data/factor/day/alpha158')
handler.load_data()
```

### 从 OptimizedDataHandler 迁移

**旧代码：**
```python
from quant_framework.data.optimized_data_handler import OptimizedDataHandler

handler = OptimizedDataHandler(
    data_path='data/factor/day/alpha158_parquet',
    use_parquet=True,
    num_workers=8
)
handler.load_data()
```

**新代码：**
```python
from quant_framework.data.data_handler import DataHandler

# 只需要改类名，其他完全相同
handler = DataHandler(
    data_path='data/factor/day/alpha158_parquet',
    use_parquet=True,
    num_workers=8
)
handler.load_data()
```

## 自动检测逻辑

当 `use_parquet=None`（默认）时：

1. 如果目录只有 `.parquet` 文件 → 使用 Parquet
2. 如果目录只有 `.csv` 文件 → 使用 CSV
3. 如果同时有两种文件 → 优先使用 Parquet
4. 如果都没有 → 抛出错误

## 常见问题

### Q: 什么时候使用多进程？

**A:**
- ✅ 数据文件很多（>100个）
- ✅ 使用 Parquet 格式
- ✅ CPU核心较多（>4核）
- ❌ 内存有限（多进程会占用更多内存）

### Q: num_workers 设置多少合适？

**A:**
- CPU密集型任务：设置为 CPU核心数
- IO密集型任务（磁盘读取）：可以设置为 CPU核心数的 1-2倍
- 保守方案：`max(1, cpu_count() - 1)`（保留一个核心）

### Q: 如何获得最佳性能？

**A:**
1. 转换为 Parquet 格式（`scripts/convert_to_parquet.py`）
2. 使用多进程（`num_workers=8` 或更多）
3. 使用股票白名单（只加载需要的股票）
4. 使用日期过滤（只加载需要的日期范围）

## 完整示例

```python
from quant_framework.data.data_handler import DataHandler

# 创建处理器（自动检测格式 + 8进程并行）
handler = DataHandler(
    data_path='data/factor/day/alpha158',
    num_workers=8
)

# 加载数据
handler.load_data(
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# 使用数据
stock_data = handler.get_stock_data('000001')
daily_data = handler.get_daily_data(date(2023, 1, 3))
all_codes = handler.get_all_codes()

# 查看信息
print(handler.get_data_info())
# 输出: DataHandler(股票: 3047, 日期: 2020-01-02 至 2023-12-29, 交易日: 975)
```

## 性能测试

运行性能对比测试：

```bash
# 转换数据为 Parquet
python scripts/convert_to_parquet.py --data-dir data/factor/day/alpha158

# 运行性能测试 + 数据验证
python scripts/benchmark_data_loading.py \
    --csv-dir data/factor/day/alpha158 \
    --parquet-dir data/factor/day/alpha158_parquet \
    --workers 8 \
    --validate
```

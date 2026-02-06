#!/usr/bin/env python3
"""
将CSV/Parquet格式的股票数据迁移到DuckDB数据库

用法:
    # 迁移股票行情数据
    python scripts/migrate_to_duckdb.py \
        --source data/stock/day \
        --target data/stock.db \
        --table stock_prices

    # 迁移Alpha158因子数据
    python scripts/migrate_to_duckdb.py \
        --source data/factor/day/alpha158 \
        --target data/factor.db \
        --table alpha158_factors \
        --workers 8
"""

import argparse
import os
import glob
from typing import List, Optional
from datetime import datetime
import pandas as pd
import duckdb
from tqdm import tqdm


def create_stock_prices_table(con: duckdb.DuckDBPyConnection, sample_df: pd.DataFrame = None) -> None:
    """
    创建股票行情表结构

    Args:
        con: DuckDB连接对象
        sample_df: 样本数据，用于动态创建列
    """
    # 基础列定义
    column_definitions = {
        'code': 'VARCHAR NOT NULL',
        'date': 'DATE NOT NULL',
        'open': 'DOUBLE',
        'high': 'DOUBLE',
        'low': 'DOUBLE',
        'close': 'DOUBLE',
        'volume': 'DOUBLE',
        'amount': 'DOUBLE',
        'outstanding_share': 'DOUBLE',
        'turnover': 'DOUBLE'
    }

    # 如果提供了样本数据，只创建样本数据中存在的列
    if sample_df is not None:
        columns_to_create = ['code', 'date']  # 必需列
        for col in sample_df.columns:
            if col in column_definitions and col not in ['code', 'date']:
                columns_to_create.append(col)

        columns_str = ', '.join([f"{col} {column_definitions[col]}" for col in columns_to_create])
    else:
        # 创建所有列
        columns_str = ', '.join([f"{col} {dtype}" for col, dtype in column_definitions.items()])

    # 创建表
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS stock_prices (
            {columns_str},
            PRIMARY KEY (code, date)
        )
    """)

    # 创建索引以加速查询
    con.execute("CREATE INDEX IF NOT EXISTS idx_code ON stock_prices(code)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_date ON stock_prices(date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_code_date ON stock_prices(code, date)")

    print("✓ 创建表: stock_prices")


def create_alpha158_factors_table(con: duckdb.DuckDBPyConnection) -> None:
    """
    创建Alpha158因子表结构

    Args:
        con: DuckDB连接对象
    """
    # Alpha158因子列名(根据实际的因子列表生成)
    # 这里先创建基础结构,实际列名会在导入时动态添加
    con.execute("""
        CREATE TABLE IF NOT EXISTS alpha158_factors (
            code VARCHAR NOT NULL,
            date DATE NOT NULL,
            PRIMARY KEY (code, date)
        )
    """)

    con.execute("CREATE INDEX IF NOT EXISTS idx_code ON alpha158_factors(code)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_date ON alpha158_factors(date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_code_date ON alpha158_factors(code, date)")

    print("✓ 创建表: alpha158_factors")


def create_trading_dates_table(con: duckdb.DuckDBPyConnection) -> None:
    """
    创建交易日历表

    Args:
        con: DuckDB连接对象
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS trading_dates (
            date DATE PRIMARY KEY,
            is_trading_day BOOLEAN NOT NULL DEFAULT TRUE
        )
    """)

    print("✓ 创建表: trading_dates")


def migrate_stock_data(source_dir: str, target_db: str, table_name: str = 'stock_prices',
                       workers: int = 1, start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> None:
    """
    迁移股票数据到DuckDB

    Args:
        source_dir: 源数据目录(CSV/Parquet文件)
        target_db: 目标DuckDB数据库文件路径
        table_name: 目标表名
        workers: 并行工作进程数
        start_date: 开始日期过滤(YYYY-MM-DD)
        end_date: 结束日期过滤(YYYY-MM-DD)
    """
    print(f"\n{'='*60}")
    print(f"开始迁移数据到DuckDB")
    print(f"{'='*60}")
    print(f"源目录: {source_dir}")
    print(f"目标数据库: {target_db}")
    print(f"目标表: {table_name}")
    print(f"并行进程数: {workers}")
    print(f"{'='*60}\n")

    # 连接到DuckDB
    con = duckdb.connect(target_db)

    # 检测文件格式
    parquet_files = glob.glob(os.path.join(source_dir, "*.parquet"))
    csv_files = glob.glob(os.path.join(source_dir, "*.csv"))

    use_parquet = len(parquet_files) > 0
    all_files = parquet_files if use_parquet else csv_files

    if not all_files:
        raise ValueError(f"在目录 {source_dir} 中未找到数据文件")

    print(f"找到 {len(all_files)} 个{'Parquet' if use_parquet else 'CSV'}文件")

    # 读取一个样本文件来检测列
    sample_df = None
    if all_files:
        try:
            if use_parquet:
                sample_df = pd.read_parquet(all_files[0])
            else:
                sample_df = pd.read_csv(all_files[0], encoding='utf-8')

            # 重命名列(如果是中文列名)
            if '股票代码' in sample_df.columns:
                sample_df = sample_df.rename(columns={
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
                })

            # 确保code列是字符串
            if 'code' not in sample_df.columns:
                code = os.path.basename(all_files[0]).replace('.parquet', '').replace('.csv', '')
                sample_df['code'] = code
            else:
                sample_df['code'] = sample_df['code'].astype(str)

            print(f"✓ 检测到列: {', '.join(sample_df.columns)}")
        except Exception as e:
            print(f"警告: 无法读取样本文件检测列: {e}")

    # 创建表结构（根据样本数据动态创建）
    create_stock_prices_table(con, sample_df)

    # 迁移数据
    print(f"\n开始导入数据...")
    start_time = datetime.now()

    success_count = 0
    error_count = 0
    total_rows = 0

    for file_path in tqdm(all_files, desc="导入进度"):
        try:
            # 读取文件
            if use_parquet:
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, encoding='utf-8')

            # 重命名列(如果是中文列名)
            if '股票代码' in df.columns:
                df = df.rename(columns={
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
                })

            # 确保code列是字符串
            if 'code' not in df.columns:
                # 从文件名提取股票代码
                code = os.path.basename(file_path).replace('.parquet', '').replace('.csv', '')
                df['code'] = code
            else:
                df['code'] = df['code'].astype(str)

            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])

            # 日期过滤
            if start_date:
                df = df[df['date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['date'] <= pd.to_datetime(end_date)]

            # 只保留需要的列
            required_columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
            optional_columns = ['amount', 'amplitude', 'pct_change', 'change_amount', 'turnover']

            # 获取表中实际存在的列（避免列不匹配错误）
            try:
                table_columns = con.execute(f"""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = '{table_name}'
                """).fetchdf()['column_name'].tolist()
            except:
                # 如果查询失败，使用默认列
                table_columns = required_columns + optional_columns

            # 只保留DataFrame中存在且表中也有列
            columns_to_keep = [col for col in df.columns if col in table_columns]
            if not columns_to_keep:
                print(f"\n警告: 文件 {file_path} 没有匹配的列，跳过")
                continue

            df = df[columns_to_keep]

            # 插入数据库（明确指定列名，避免列数不匹配）
            if len(df) > 0:  # 确保有数据
                con.register('temp_df', df)
                columns_str = ', '.join(df.columns)
                con.execute(f"INSERT OR REPLACE INTO {table_name} ({columns_str}) SELECT {columns_str} FROM temp_df")
                con.unregister('temp_df')

            success_count += 1
            total_rows += len(df)

        except Exception as e:
            error_count += 1
            print(f"\n错误: 处理文件 {file_path} 失败: {e}")
            continue

    # 创建交易日历表
    print("\n创建交易日历...")
    create_trading_dates_table(con)

    # 从数据中提取所有交易日期
    con.execute("""
        INSERT OR REPLACE INTO trading_dates (date, is_trading_day)
        SELECT DISTINCT date, TRUE FROM stock_prices
    """)

    # 打印统计信息
    elapsed_time = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*60}")
    print(f"迁移完成!")
    print(f"{'='*60}")
    print(f"成功导入: {success_count} 个文件")
    print(f"失败: {error_count} 个文件")
    print(f"总行数: {total_rows:,}")
    print(f"耗时: {elapsed_time:.2f}秒")
    print(f"速度: {total_rows/elapsed_time:,.0f} 行/秒")
    print(f"{'='*60}\n")

    # 查询数据库统计信息
    result = con.execute(f"""
        SELECT
            COUNT(DISTINCT code) as stock_count,
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(*) as total_records
        FROM {table_name}
    """).fetchdf()

    print("\n数据库统计:")
    print(result.to_string(index=False))

    # 获取数据库文件大小
    if os.path.exists(target_db):
        db_size_mb = os.path.getsize(target_db) / (1024 * 1024)
        print(f"\n数据库文件大小: {db_size_mb:.2f} MB")

    con.close()


def migrate_factor_data(source_dir: str, target_db: str, table_name: str = 'alpha158_factors',
                        workers: int = 1) -> None:
    """
    迁移Alpha158因子数据到DuckDB

    Args:
        source_dir: 源数据目录(CSV/Parquet文件)
        target_db: 目标DuckDB数据库文件路径
        table_name: 目标表名
        workers: 并行工作进程数
    """
    print(f"\n{'='*60}")
    print(f"开始迁移Alpha158因子数据到DuckDB")
    print(f"{'='*60}")
    print(f"源目录: {source_dir}")
    print(f"目标数据库: {target_db}")
    print(f"目标表: {table_name}")
    print(f"{'='*60}\n")

    # 连接到DuckDB
    con = duckdb.connect(target_db)

    # 检测文件格式
    parquet_files = glob.glob(os.path.join(source_dir, "*.parquet"))
    csv_files = glob.glob(os.path.join(source_dir, "*.csv"))

    use_parquet = len(parquet_files) > 0
    all_files = parquet_files if use_parquet else csv_files

    if not all_files:
        raise ValueError(f"在目录 {source_dir} 中未找到数据文件")

    print(f"找到 {len(all_files)} 个{'Parquet' if use_parquet else 'CSV'}文件")

    # 先读取一个文件来获取所有列名
    sample_file = all_files[0]
    if use_parquet:
        sample_df = pd.read_parquet(sample_file)
    else:
        sample_df = pd.read_csv(sample_file, encoding='utf-8')

    # 重命名列(如果是中文列名)
    if '股票代码' in sample_df.columns:
        column_mapping = {
            '日期': 'date',
            '股票代码': 'code',
            **{f'因子{i}': f'factor_{i}' for i in range(1, 159)}  # 示例映射
        }
        sample_df = sample_df.rename(columns=column_mapping)

    # 确保code列是字符串
    if 'code' not in sample_df.columns:
        code = os.path.basename(sample_file).replace('.parquet', '').replace('.csv', '')
        sample_df['code'] = code
    else:
        sample_df['code'] = sample_df['code'].astype(str)

    print(f"✓ 检测到 {len(sample_df.columns)} 列: {', '.join(list(sample_df.columns[:10]))}...")

    # 创建表结构(动态添加所有因子列)
    columns_def = ["code VARCHAR NOT NULL", "date DATE NOT NULL"]
    for col in sample_df.columns:
        if col not in ['code', 'date']:
            columns_def.append(f"{col} DOUBLE")

    # 删除旧表（如果存在）
    con.execute(f"DROP TABLE IF EXISTS {table_name}")

    con.execute(f"""
        CREATE TABLE {table_name} (
            {', '.join(columns_def)},
            PRIMARY KEY (code, date)
        )
    """)

    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_code ON {table_name}(code)")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_code_date ON {table_name}(code, date)")

    print(f"✓ 创建表: {table_name} (包含 {len(columns_def)} 列)")

    # 迁移数据
    print(f"\n开始导入数据...")
    start_time = datetime.now()

    success_count = 0
    error_count = 0
    total_rows = 0

    for file_path in tqdm(all_files, desc="导入进度"):
        try:
            # 读取文件
            if use_parquet:
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, encoding='utf-8')

            # 重命名列(如果是中文列名)
            if '股票代码' in df.columns:
                df = df.rename(columns=column_mapping)

            # 确保code列是字符串
            if 'code' not in df.columns:
                code = os.path.basename(file_path).replace('.parquet', '').replace('.csv', '')
                df['code'] = code
            else:
                df['code'] = df['code'].astype(str)

            # 转换日期格式
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            # 获取表中实际存在的列（避免列不匹配错误）
            try:
                table_columns = con.execute(f"""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = '{table_name}'
                """).fetchdf()['column_name'].tolist()
            except:
                # 如果查询失败，使用DataFrame中的所有列
                table_columns = df.columns.tolist()

            # 只保留DataFrame中存在且表中也有列
            columns_to_keep = [col for col in df.columns if col in table_columns]
            if not columns_to_keep:
                print(f"\n警告: 文件 {file_path} 没有匹配的列，跳过")
                continue

            df = df[columns_to_keep]

            # 插入数据库（明确指定列名，避免列数不匹配）
            if len(df) > 0:  # 确保有数据
                con.register('temp_df', df)
                columns_str = ', '.join(df.columns)
                con.execute(f"INSERT OR REPLACE INTO {table_name} ({columns_str}) SELECT {columns_str} FROM temp_df")
                con.unregister('temp_df')

            success_count += 1
            total_rows += len(df)

        except Exception as e:
            error_count += 1
            print(f"\n错误: 处理文件 {file_path} 失败: {e}")
            continue

    # 打印统计信息
    elapsed_time = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*60}")
    print(f"迁移完成!")
    print(f"{'='*60}")
    print(f"成功导入: {success_count} 个文件")
    print(f"失败: {error_count} 个文件")
    print(f"总行数: {total_rows:,}")
    print(f"耗时: {elapsed_time:.2f}秒")
    print(f"速度: {total_rows/elapsed_time:,.0f} 行/秒")
    print(f"{'='*60}\n")

    # 查询数据库统计信息
    result = con.execute(f"""
        SELECT
            COUNT(DISTINCT code) as stock_count,
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(*) as total_records
        FROM {table_name}
    """).fetchdf()

    print("\n数据库统计:")
    print(result.to_string(index=False))

    con.close()


def main():
    parser = argparse.ArgumentParser(description='迁移股票数据到DuckDB数据库')
    parser.add_argument('--source', required=True, help='源数据目录')
    parser.add_argument('--target', required=True, help='目标DuckDB数据库文件路径')
    parser.add_argument('--table', default='stock_prices', help='目标表名')
    parser.add_argument('--type', choices=['stock', 'factor'], default='stock',
                        help='数据类型: stock=股票行情, factor=Alpha158因子')
    parser.add_argument('--workers', type=int, default=1, help='并行工作进程数')
    parser.add_argument('--start-date', help='开始日期(YYYY-MM-DD)')
    parser.add_argument('--end-date', help='结束日期(YYYY-MM-DD)')

    args = parser.parse_args()

    if args.type == 'stock':
        migrate_stock_data(
            source_dir=args.source,
            target_db=args.target,
            table_name=args.table,
            workers=args.workers,
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:
        migrate_factor_data(
            source_dir=args.source,
            target_db=args.target,
            table_name=args.table,
            workers=args.workers
        )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
DuckDB 数据库使用示例

演示如何使用DatabaseDataHandler进行数据查询
"""

from datetime import date
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_framework.data.database_data_handler import DatabaseDataHandler, FactorDataHandler


def example_basic_usage():
    """基本使用示例"""
    print("\n" + "="*60)
    print("示例1: 基本使用")
    print("="*60)

    # 创建数据库数据处理器
    # 注意: 需要先运行迁移脚本创建数据库
    # python scripts/migrate_to_duckdb.py --source data/stock/day --target data/stock.db

    try:
        handler = DatabaseDataHandler(
            db_path='data/stock.db',
            table_name='stock_prices'
        )

        print("\n✓ 数据库连接成功")
        print(f"  - {handler}")

        # 获取单只股票数据
        print("\n--- 获取单只股票数据 ---")
        df = handler.get_stock_data('000001')
        print(f"股票 000001 的数据:")
        print(df.head())

        # 获取指定日期的所有股票数据
        print("\n--- 获取截面数据 ---")
        df_daily = handler.get_daily_data(date(2024, 1, 10))
        print(f"2024-01-10 的股票数量: {len(df_daily)}")
        print(df_daily.head())

        # 获取交易日历
        print("\n--- 获取交易日历 ---")
        dates = handler.get_available_dates(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        print(f"2024年1月的交易日数量: {len(dates)}")
        print(f"前5个交易日: {dates[:5]}")

        handler.close()

    except FileNotFoundError as e:
        print(f"\n✗ 数据库文件不存在: {e}")
        print("  请先运行迁移脚本: python scripts/migrate_to_duckdb.py")


def example_batch_query():
    """批量查询示例"""
    print("\n" + "="*60)
    print("示例2: 批量查询(高性能)")
    print("="*60)

    try:
        handler = DatabaseDataHandler(
            db_path='data/stock.db',
            table_name='stock_prices'
        )

        print("\n--- 批量获取多只股票的数据 ---")
        codes = handler.get_all_codes()[:10]  # 获取前10只股票
        print(f"查询股票: {codes[:5]}...")

        df = handler.get_data_range(
            codes=codes,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )

        print(f"✓ 查询成功")
        print(f"  - 股票数量: {df['code'].nunique()}")
        print(f"  - 数据行数: {len(df):,}")
        print(df.head(10))

        handler.close()

    except FileNotFoundError as e:
        print(f"\n✗ 数据库文件不存在: {e}")


def example_sql_query():
    """SQL查询示例"""
    print("\n" + "="*60)
    print("示例3: SQL查询")
    print("="*60)

    try:
        handler = DatabaseDataHandler(
            db_path='data/stock.db',
            table_name='stock_prices'
        )

        # 查询涨幅最大的股票
        print("\n--- 查询2024年涨幅TOP10股票 ---")
        sql = """
        SELECT code,
               (LAST(close) - FIRST(close)) / FIRST(close) as return_rate,
               FIRST(close) as start_price,
               LAST(close) as end_price
        FROM stock_prices
        WHERE date >= '2024-01-01' AND date <= '2024-12-31'
        GROUP BY code
        ORDER BY return_rate DESC
        LIMIT 10
        """

        result = handler.execute_sql(sql)
        print(result.to_string(index=False))

        # 查询成交额最大的股票
        print("\n--- 查询2024年成交额TOP10股票 ---")
        sql = """
        SELECT code,
               SUM(amount) / 100000000 as total_amount_yi
        FROM stock_prices
        WHERE date >= '2024-01-01' AND date <= '2024-12-31'
        GROUP BY code
        ORDER BY total_amount_yi DESC
        LIMIT 10
        """

        result = handler.execute_sql(sql)
        print(result.to_string(index=False))

        handler.close()

    except FileNotFoundError as e:
        print(f"\n✗ 数据库文件不存在: {e}")


def example_factor_query():
    """因子数据查询示例"""
    print("\n" + "="*60)
    print("示例4: 因子数据查询")
    print("="*60)

    try:
        factor_handler = FactorDataHandler(
            db_path='data/factor.db',
            table_name='alpha158_factors'
        )

        print("\n✓ 因子数据库连接成功")
        print(f"  - {factor_handler}")

        # 获取可用的因子列名
        print("\n--- 获取可用因子列 ---")
        factor_columns = factor_handler.get_available_factors()
        print(f"因子数量: {len(factor_columns)}")
        print(f"前10个因子: {factor_columns[:10]}")

        # 获取截面因子数据
        print("\n--- 获取截面因子数据 ---")
        cross_section = factor_handler.get_factor_cross_section(
            date=date(2024, 1, 10)
        )
        print(f"2024-01-10 的股票数量: {len(cross_section)}")
        print(cross_section.head())

        # 批量获取因子数据
        print("\n--- 批量获取因子数据 ---")
        codes = factor_handler.get_all_codes()[:5]
        print(f"查询股票: {codes}")

        factors = factor_handler.get_factors(
            codes=codes,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            factor_columns=factor_columns[:5]  # 只获取前5个因子
        )

        print(f"✓ 查询成功")
        print(f"  - 数据行数: {len(factors)}")
        print(factors.head())

        factor_handler.close()

    except FileNotFoundError as e:
        print(f"\n✗ 因子数据库文件不存在: {e}")
        print("  请先运行迁移脚本: python scripts/migrate_to_duckdb.py --type factor")


def example_trading_calendar():
    """交易日历查询示例"""
    print("\n" + "="*60)
    print("示例5: 交易日历查询")
    print("="*60)

    try:
        handler = DatabaseDataHandler(
            db_path='data/stock.db',
            table_name='stock_prices'
        )

        # 获取前一个交易日
        print("\n--- 获取前一个交易日 ---")
        current_date = date(2024, 1, 10)
        prev_date = handler.get_previous_trading_date(current_date, n=1)
        prev_5_date = handler.get_previous_trading_date(current_date, n=5)

        print(f"当前日期: {current_date}")
        print(f"前1个交易日: {prev_date}")
        print(f"前5个交易日: {prev_5_date}")

        handler.close()

    except FileNotFoundError as e:
        print(f"\n✗ 数据库文件不存在: {e}")


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("DuckDB 数据库使用示例")
    print("="*60)

    # 运行示例
    example_basic_usage()
    example_batch_query()
    example_sql_query()
    example_factor_query()
    example_trading_calendar()

    print("\n" + "="*60)
    print("所有示例运行完成!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

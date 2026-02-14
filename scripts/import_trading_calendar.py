#!/usr/bin/env python3
"""
导入交易日历到数据库

将 trading_calendar.csv 文件中的交易日数据导入到 DuckDB 数据库的 trading_dates 表中。
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import duckdb

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def import_trading_calendar(csv_path: str, db_path: str, force: bool = False) -> None:
    """
    导入交易日历到数据库

    Args:
        csv_path: CSV 文件路径
        db_path: 数据库文件路径
        force: 是否强制清空现有数据（默认需要确认）
    """
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    print(f"正在读取 CSV 文件: {csv_path}")
    df = pd.read_csv(csv_path)

    # 检查 CSV 格式
    if 'trade_date' not in df.columns:
        raise ValueError("CSV 文件必须包含 'trade_date' 列")

    # 转换数据格式
    df['date'] = pd.to_datetime(df['trade_date'])
    df['is_trading_day'] = True

    # 只保留需要的列
    df = df[['date', 'is_trading_day']]

    print(f"✓ 读取成功，共 {len(df)} 条交易日记录")
    print(f"  日期范围: {df['date'].min().date()} 至 {df['date'].max().date()}")

    # 连接数据库
    print(f"\n正在连接数据库: {db_path}")
    con = duckdb.connect(db_path)

    # 检查表是否存在
    table_exists = con.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_name = 'trading_dates'
    """).fetchone()

    if not table_exists:
        print("⚠ 警告: trading_dates 表不存在，将创建该表")

        # 创建表
        con.execute("""
            CREATE TABLE trading_dates (
                date DATE PRIMARY KEY,
                is_trading_day BOOLEAN NOT NULL DEFAULT TRUE
            )
        """)
        print("✓ 表创建成功")
    else:
        # 检查现有数据
        existing_count = con.execute("SELECT COUNT(*) FROM trading_dates").fetchone()[0]

        if existing_count > 0:
            if not force:
                response = input(f"\n表中现有 {existing_count} 条数据，确认清空并重新导入？(yes/no): ")
                if response.lower() not in ['yes', 'y']:
                    print("✗ 操作已取消")
                    return

            print(f"正在清空现有数据 ({existing_count} 条)...")
            con.execute("DELETE FROM trading_dates")
            print("✓ 清空完成")

    # 批量导入数据
    print(f"\n正在导入 {len(df)} 条交易日记录...")

    # 注册临时表并插入
    con.register('temp_trading_dates', df)
    con.execute("""
        INSERT INTO trading_dates
        SELECT date, is_trading_day
        FROM temp_trading_dates
    """)
    con.unregister('temp_trading_dates')

    print("✓ 导入完成")

    # 验证结果
    print("\n正在验证导入结果...")

    result = con.execute("""
        SELECT
            COUNT(*) as total_count,
            MIN(date) as min_date,
            MAX(date) as max_date,
            SUM(CASE WHEN is_trading_day THEN 1 ELSE 0 END) as trading_day_count
        FROM trading_dates
    """).fetchdf()

    row = result.iloc[0]

    print(f"\n{'='*60}")
    print("导入结果统计:")
    print(f"{'='*60}")
    print(f"总记录数:     {row['total_count']:,}")
    print(f"交易日数:     {row['trading_day_count']:,}")
    print(f"日期范围:     {row['min_date']} 至 {row['max_date']}")
    print(f"{'='*60}")

    # 显示前5条和后5条记录
    print("\n最早5个交易日:")
    earliest = con.execute("SELECT * FROM trading_dates ORDER BY date ASC LIMIT 5").fetchdf()
    print(earliest.to_string(index=False))

    print("\n最晚5个交易日:")
    latest = con.execute("SELECT * FROM trading_dates ORDER BY date DESC LIMIT 5").fetchdf()
    print(latest.to_string(index=False))

    # 关闭连接
    con.close()

    print(f"\n✓ 交易日历导入成功!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='导入交易日历到数据库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认路径导入
  python scripts/import_trading_calendar.py

  # 指定数据库和CSV路径
  python scripts/import_trading_calendar.py --db-path data/stock.db --csv-path data/calendar/trading_calendar.csv

  # 强制导入（不询问确认）
  python scripts/import_trading_calendar.py --force
        """
    )

    parser.add_argument(
        '--csv-path',
        type=str,
        default='data/calendar/trading_calendar.csv',
        help='交易日历CSV文件路径 (默认: data/calendar/trading_calendar.csv)'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='data/stock.db',
        help='数据库文件路径 (默认: data/stock.db)'
    )

    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='强制导入，不询问确认'
    )

    args = parser.parse_args()

    try:
        import_trading_calendar(
            csv_path=args.csv_path,
            db_path=args.db_path,
            force=args.force
        )
    except Exception as e:
        print(f"\n✗ 导入失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

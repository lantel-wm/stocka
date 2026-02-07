#!/usr/bin/env python3
"""
初始化因子表结构

在现有的 stock.db 中创建因子相关表，并注册 Alpha158 因子定义

用法:
    python scripts/init_factor_tables.py --db-path data/stock.db
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_framework.data.data_handler import DataHandler
from quant_framework.factor.alpha158 import Alpha158


def init_factor_tables(db_path: str, force: bool = False):
    """
    初始化因子表结构

    Args:
        db_path: 数据库文件路径
        force: 是否强制重新创建表（会删除现有数据）
    """
    print(f"\n{'='*60}")
    print(f"初始化因子表结构")
    print(f"{'='*60}")
    print(f"数据库: {db_path}")
    print(f"{'='*60}\n")

    # 连接数据库
    handler = DataHandler(db_path, table_name='stock_prices')

    # 如果强制重建，先删除因子表
    if force:
        print("⚠️  强制模式：删除现有因子表...")
        try:
            handler.con.execute("DROP TABLE IF EXISTS factor_data")
            handler.con.execute("DROP TABLE IF EXISTS factor_definitions")
            print("✓ 已删除现有因子表")
        except Exception as e:
            print(f"✗ 删除因子表失败: {e}")
            handler.close()
            return

    # 初始化因子表
    print("\n[1/2] 创建因子表结构...")
    try:
        handler._init_factor_tables()
        print("✓ 因子表结构创建成功")
    except Exception as e:
        print(f"✗ 创建因子表失败: {e}")
        handler.close()
        return

    # 注册 Alpha158 因子定义
    print(f"\n[2/2] 注册 Alpha158 因子定义...")
    registered_count = 0
    skipped_count = 0
    error_count = 0

    for factor_id, factor_name, category, desc in Alpha158.DEFINITIONS:
        try:
            # 使用 on_conflict='skip' 跳过已存在的因子
            result_id = handler.register_factor(factor_id, factor_name, category, desc, on_conflict='skip')
            if result_id == factor_id:
                registered_count += 1
                if registered_count <= 10 or registered_count % 30 == 0:
                    print(f"  注册因子: {factor_name} (ID: {factor_id})")
            else:
                skipped_count += 1
        except Exception as e:
            error_count += 1
            print(f"  ✗ 注册因子 {factor_name} (ID: {factor_id}) 失败: {e}")

    print(f"\n✓ 成功注册 {registered_count} 个因子定义")
    if skipped_count > 0:
        print(f"  (跳过 {skipped_count} 个已存在的因子)")
    if error_count > 0:
        print(f"  ✗ {error_count} 个因子注册失败")

    # 查询并显示因子定义信息
    print(f"\n{'='*60}")
    print(f"因子表初始化完成!")
    print(f"{'='*60}")

    factor_info = handler.get_factor_info()
    print(f"\n因子定义总数: {len(factor_info)}")
    print(f"\n按类别统计:")
    print(factor_info.groupby('factor_category').size())

    print(f"\n前10个因子:")
    print(factor_info[['factor_id', 'factor_name', 'factor_category']].head(10).to_string(index=False))

    # 查询表信息
    print(f"\n数据库表列表:")
    tables = handler.con.execute("SHOW TABLES").fetchdf()
    print(tables.to_string(index=False))

    handler.close()


def main():
    parser = argparse.ArgumentParser(description='初始化因子表结构')
    parser.add_argument('--db-path', default='data/stock.db',
                       help='数据库文件路径 (默认: data/stock.db)')
    parser.add_argument('--force', action='store_true',
                       help='强制重新创建表（会删除现有数据）')

    args = parser.parse_args()

    # 检查数据库文件是否存在
    if not Path(args.db_path).exists():
        print(f"✗ 错误: 数据库文件不存在: {args.db_path}")
        print("  请先运行迁移脚本创建数据库: python scripts/migrate_to_duckdb.py --source data/stock/day --target data/stock.db")
        sys.exit(1)

    init_factor_tables(args.db_path, args.force)


if __name__ == '__main__':
    main()

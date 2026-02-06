#!/usr/bin/env python3
"""
将 Parquet 格式的因子数据迁移到 DuckDB 数据库

支持将 data/factor/day/alpha158/*.parquet 文件迁移到 stock.db 的 factor_data 表

用法:
    python scripts/migrate_parquet_factors.py \
        --source data/factor/day/alpha158 \
        --db data/stock.db 
"""

import argparse
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_framework.data.data_handler import DataHandler


# Alpha158 因子定义（与 alpha158.py 和 init_factor_tables.py 保持一致）
ALPHA158_FACTORS = [
    # KBar 因子
    ('KMID', 'alpha158', '收盘价与开盘价的差值除以收盘价'),
    ('KLEN', 'alpha158', '最高价与最低价的差值除以开盘价'),
    ('KMID2', 'alpha158', '收盘价与开盘价的差值除以振幅'),
    ('KUP', 'alpha158', '最高价与（开盘价、收盘价较大值）的差值除以开盘价'),
    ('KUP2', 'alpha158', '最高价与（开盘价、收盘价较大值）的差值除以振幅'),
    ('KLOW', 'alpha158', '（开盘价、收盘价较小值）与最低价的差值除以开盘价'),
    ('KLOW2', 'alpha158', '（开盘价、收盘价较小值）与最低价的差值除以振幅'),
    ('KSFT', 'alpha158', '2倍收盘价减去最高价和最低价的差值除以开盘价'),
    ('KSFT2', 'alpha158', '2倍收盘价减去最高价和最低价的差值除以振幅'),

    # Price 因子
    ('OPEN0', 'alpha158', '开盘价除以收盘价'),
    ('HIGH0', 'alpha158', '最高价除以收盘价'),
    ('LOW0', 'alpha158', '最低价除以收盘价'),

    # ROC 因子 (5个窗口)
    *[(f'ROC{w}', 'alpha158', f'{w}日收益率') for w in [5, 10, 20, 30, 60]],

    # MA 因子 (5个窗口)
    *[(f'MA{w}', 'alpha158', f'{w}日移动平均除以收盘价') for w in [5, 10, 20, 30, 60]],

    # STD 因子 (5个窗口)
    *[(f'STD{w}', 'alpha158', f'{w}日标准差除以收盘价') for w in [5, 10, 20, 30, 60]],

    # BETA 因子 (5个窗口)
    *[(f'BETA{w}', 'alpha158', f'{w}日线性回归斜率除以收盘价') for w in [5, 10, 20, 30, 60]],

    # RSQR 因子 (5个窗口)
    *[(f'RSQR{w}', 'alpha158', f'{w}日线性回归R平方') for w in [5, 10, 20, 30, 60]],

    # RESI 因子 (5个窗口)
    *[(f'RESI{w}', 'alpha158', f'{w}日线性回归残差除以收盘价') for w in [5, 10, 20, 30, 60]],

    # MAX 因子 (5个窗口)
    *[(f'MAX{w}', 'alpha158', f'{w}日最高价除以收盘价') for w in [5, 10, 20, 30, 60]],

    # MIN 因子 (5个窗口)
    *[(f'MIN{w}', 'alpha158', f'{w}日最低价除以收盘价') for w in [5, 10, 20, 30, 60]],

    # QTLU 因子 (5个窗口)
    *[(f'QTLU{w}', 'alpha158', f'{w}日80%分位数除以收盘价') for w in [5, 10, 20, 30, 60]],

    # QTLD 因子 (5个窗口)
    *[(f'QTLD{w}', 'alpha158', f'{w}日20%分位数除以收盘价') for w in [5, 10, 20, 30, 60]],

    # RSV 因子 (5个窗口)
    *[(f'RSV{w}', 'alpha158', f'{w}日RSV') for w in [5, 10, 20, 30, 60]],

    # IMAX 因子 (5个窗口)
    *[(f'IMAX{w}', 'alpha158', f'{w}日内最高价位置除以窗口长度') for w in [5, 10, 20, 30, 60]],

    # IMIN 因子 (5个窗口)
    *[(f'IMIN{w}', 'alpha158', f'{w}日内最低价位置除以窗口长度') for w in [5, 10, 20, 30, 60]],

    # IMXD 因子 (5个窗口)
    *[(f'IMXD{w}', 'alpha158', f'{w}日内最高价与最低价位置之差除以窗口长度') for w in [5, 10, 20, 30, 60]],

    # CORR 因子 (5个窗口)
    *[(f'CORR{w}', 'alpha158', f'{w}日收盘价与成交量的相关系数') for w in [5, 10, 20, 30, 60]],

    # CORD 因子 (5个窗口)
    *[(f'CORD{w}', 'alpha158', f'{w}日价格变化率与成交量变化率的相关系数') for w in [5, 10, 20, 30, 60]],

    # CNTP 因子 (5个窗口)
    *[(f'CNTP{w}', 'alpha158', f'{w}日内上涨天数占比') for w in [5, 10, 20, 30, 60]],

    # CNTN 因子 (5个窗口)
    *[(f'CNTN{w}', 'alpha158', f'{w}日内下跌天数占比') for w in [5, 10, 20, 30, 60]],

    # CNTD 因子 (5个窗口)
    *[(f'CNTD{w}', 'alpha158', f'{w}日内上涨天数占比与下跌天数占比之差') for w in [5, 10, 20, 30, 60]],

    # SUMP 因子 (5个窗口)
    *[(f'SUMP{w}', 'alpha158', f'{w}日内上涨幅度之和除以总变化幅度') for w in [5, 10, 20, 30, 60]],

    # SUMN 因子 (5个窗口)
    *[(f'SUMN{w}', 'alpha158', f'{w}日内下跌幅度之和除以总变化幅度') for w in [5, 10, 20, 30, 60]],

    # SUMD 因子 (5个窗口)
    *[(f'SUMD{w}', 'alpha158', f'{w}日内上涨与下跌幅度之差除以总变化幅度') for w in [5, 10, 20, 30, 60]],

    # VMA 因子 (5个窗口)
    *[(f'VMA{w}', 'alpha158', f'{w}日平均成交量除以当日成交量') for w in [5, 10, 20, 30, 60]],

    # VSTD 因子 (5个窗口)
    *[(f'VSTD{w}', 'alpha158', f'{w}日成交量标准差除以当日成交量') for w in [5, 10, 20, 30, 60]],

    # WVMA 因子 (5个窗口)
    *[(f'WVMA{w}', 'alpha158', f'{w}日加权成交量标准差除以加权平均成交量') for w in [5, 10, 20, 30, 60]],

    # VSUMP 因子 (5个窗口)
    *[(f'VSUMP{w}', 'alpha158', f'{w}日内成交量上涨之和除以总变化') for w in [5, 10, 20, 30, 60]],

    # VSUMN 因子 (5个窗口)
    *[(f'VSUMN{w}', 'alpha158', f'{w}日内成交量下跌之和除以总变化') for w in [5, 10, 20, 30, 60]],

    # VSUMD 因子 (5个窗口)
    *[(f'VSUMD{w}', 'alpha158', f'{w}日内成交量上涨与下跌之差除以总变化') for w in [5, 10, 20, 30, 60]],
]


def migrate_factors(source_dir: str, db_path: str,
                   start_date: str = None, end_date: str = None,
                   batch_size: int = 100) -> None:
    """
    迁移因子数据到数据库

    Args:
        source_dir: 源数据目录（parquet文件）
        db_path: 目标数据库路径
        start_date: 开始日期过滤 (YYYY-MM-DD)
        end_date: 结束日期过滤 (YYYY-MM-DD)
        batch_size: 批量插入的记录数
    """
    print(f"\n{'='*60}")
    print(f"迁移因子数据到数据库")
    print(f"{'='*60}")
    print(f"源目录: {source_dir}")
    print(f"目标数据库: {db_path}")
    print(f"批量大小: {batch_size} 条记录/批")
    print(f"{'='*60}\n")

    # 检查源目录
    if not os.path.exists(source_dir):
        print(f"✗ 错误: 源目录不存在: {source_dir}")
        return

    # 查找 parquet 文件
    parquet_files = glob.glob(os.path.join(source_dir, "*.parquet"))

    if not parquet_files:
        print(f"✗ 错误: 在目录 {source_dir} 中未找到 parquet 文件")
        return

    print(f"找到 {len(parquet_files)} 个 parquet 文件\n")

    # 连接数据库
    handler = DataHandler(db_path, table_name='stock_prices')

    # 检查并修复因子表结构
    print("[1/4] 检查因子表...")

    # 检查是否需要重建表（检测旧结构）
    tables = handler.con.execute("SHOW TABLES").fetchdf()
    has_factor_definitions = 'factor_definitions' in tables['name'].values

    needs_rebuild = False
    if has_factor_definitions:
        # 检查表结构是否正确（是否有序列）
        try:
            # 尝试注册一个测试因子
            handler.con.execute("BEGIN TRANSACTION")
            try:
                handler.con.execute("""
                    INSERT INTO factor_definitions (factor_id, factor_name, factor_category, factor_desc)
                    VALUES (nextval('factor_id_seq'), '__test__', 'test', 'test')
                """)
                handler.con.execute("DELETE FROM factor_definitions WHERE factor_name = '__test__'")
                handler.con.execute("COMMIT")
            except Exception:
                handler.con.execute("ROLLBACK")
                needs_rebuild = True
        except:
            needs_rebuild = True

    if needs_rebuild:
        print("  检测到旧表结构，需要重建...")

        # 查询现有数据量
        try:
            count = handler.con.execute("SELECT COUNT(*) FROM factor_data").fetchone()[0]
            print(f"  当前有 {count:,} 条因子数据将被删除")
        except:
            count = 0

        # 删除旧表
        print("  删除旧表...")
        handler.con.execute("DROP TABLE IF EXISTS factor_data")
        handler.con.execute("DROP TABLE IF EXISTS factor_definitions")
        handler.con.execute("DROP SEQUENCE IF EXISTS factor_id_seq")
        print("  ✓ 旧表已删除\n")

    # 初始化因子表
    handler._init_factor_tables()
    print("✓ 因子表检查完成\n")

    # 注册 Alpha158 因子定义
    print(f"[2/4] 注册 Alpha158 因子定义...")
    registered_count = 0
    skipped_count = 0

    for factor_name, category, desc in ALPHA158_FACTORS:
        try:
            factor_id = handler.register_factor(factor_name, category, desc)
            if factor_id:
                registered_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"  警告: 注册因子 {factor_name} 失败: {e}")

    print(f"✓ 成功注册 {registered_count} 个因子定义")
    if skipped_count > 0:
        print(f"  (跳过 {skipped_count} 个已存在的因子)")

    # 额外：检测parquet文件中实际存在的因子并自动注册
    print(f"\n[2.5/4] 扫描parquet文件中的因子列...")
    sample_file = parquet_files[0]
    sample_df = pd.read_parquet(sample_file)

    base_columns = {'date', 'code', 'open', 'high', 'low', 'close', 'volume',
                   'amount', 'outstanding_share', 'turnover'}
    actual_factor_columns = set(sample_df.columns) - base_columns

    print(f"  发现 {len(actual_factor_columns)} 个因子列")

    # 检查是否有未注册的因子
    registered_factors = set(handler.get_available_factors())
    unregistered_factors = actual_factor_columns - registered_factors

    if unregistered_factors:
        print(f"  发现 {len(unregistered_factors)} 个未注册的因子，自动注册...")
        auto_registered = 0
        for factor_name in sorted(unregistered_factors):
            try:
                handler.register_factor(factor_name, 'alpha158', f'{factor_name} 因子')
                auto_registered += 1
                if auto_registered <= 5:
                    print(f"    自动注册: {factor_name}")
            except Exception as e:
                print(f"    警告: 注册 {factor_name} 失败: {e}")

        print(f"  ✓ 自动注册了 {auto_registered} 个因子")
    else:
        print(f"  ✓ 所有因子列都已注册")

    print()

    # 迁移数据
    print(f"\n[3/4] 开始迁移因子数据...")
    start_time = datetime.now()

    success_count = 0
    error_count = 0
    total_dates = 0
    total_records = 0

    # 进度条
    pbar = tqdm(parquet_files, desc="处理进度")

    for file_path in pbar:
        try:
            # 读取 parquet 文件
            df = pd.read_parquet(file_path)

            # 检查必要的列
            if 'date' not in df.columns or 'code' not in df.columns:
                pbar.set_postfix({"error": f"缺少必要列 {os.path.basename(file_path)}"})
                error_count += 1
                continue

            # 日期过滤
            if start_date:
                df = df[df['date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['date'] <= pd.to_datetime(end_date)]

            if len(df) == 0:
                continue

            # 获取因子列（排除基础列）
            base_columns = {'date', 'code', 'open', 'high', 'low', 'close', 'volume',
                           'amount', 'outstanding_share', 'turnover'}
            factor_columns = [col for col in df.columns if col not in base_columns]

            if not factor_columns:
                pbar.set_postfix({"error": "无因子列"})
                error_count += 1
                continue

            # 按日期分组保存
            for trade_date, group in df.groupby('date'):
                # try:
                # 准备因子数据
                factor_df = group.set_index('code')[factor_columns].copy()

                # 删除全为NaN的行
                factor_df = factor_df.dropna(how='all')

                if len(factor_df) == 0:
                    continue

                # 转换日期为 date 对象
                if isinstance(trade_date, str):
                    trade_date_obj = pd.to_datetime(trade_date).date()
                elif hasattr(trade_date, 'date'):
                    trade_date_obj = trade_date.date()
                else:
                    trade_date_obj = pd.to_datetime(trade_date).date()

                # 保存到数据库
                handler.save_factors(factor_df, trade_date_obj)

                total_dates += 1
                total_records += len(factor_df) * len(factor_columns)

                # except Exception as e:
                #     # 打印完整错误信息并中止
                #     print(f"\n{'='*60}")
                #     print(f"❌ 保存失败！")
                #     print(f"{'='*60}")
                #     print(f"文件: {file_path}")
                #     print(f"日期: {trade_date}")
                #     print(f"错误: {e}")
                #     print(f"\n因子列数: {len(factor_columns)}")
                #     print(f"股票数: {len(factor_df)}")
                #     print(f"数据形状: {factor_df.shape}")

                #     # 检查未注册的因子
                #     print(f"\n检查因子注册状态:")
                #     missing_factors = []
                #     for col in factor_columns:
                #         factor_id = handler.get_factor_id(col)
                #         if factor_id is None:
                #             missing_factors.append(col)
                #             print(f"  ✗ {col} - 未注册")
                #         elif len(missing_factors) < 5:  # 只显示前5个已注册的
                #             print(f"  ✓ {col} - ID: {factor_id}")

                #     if missing_factors:
                #         print(f"\n未注册的因子 ({len(missing_factors)} 个):")
                #         for f in missing_factors[:10]:  # 显示前10个
                #             print(f"  - {f}")
                #         if len(missing_factors) > 10:
                #             print(f"  ... 还有 {len(missing_factors) - 10} 个")

                #     # 打印完整 traceback
                #     print(f"\n完整错误堆栈:")
                #     import traceback
                #     traceback.print_exc()

                #     print(f"\n{'='*60}")
                #     print("脚本中止！请修复上述错误后重试。")
                #     print(f"{'='*60}\n")

                #     handler.close()
                #     sys.exit(1)

            success_count += 1
            pbar.set_postfix({"success": success_count, "error": error_count})

        except Exception as e:
            pbar.set_postfix({"error": f"读取失败 {os.path.basename(file_path)}"})
            error_count += 1
            continue

    # 打印统计信息
    elapsed_time = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*60}")
    print(f"迁移完成!")
    print(f"{'='*60}")
    print(f"成功处理: {success_count} 个文件")
    print(f"失败: {error_count} 个文件")
    print(f"总日期数: {total_dates:,}")
    print(f"总记录数: ~{total_records:,}")
    print(f"耗时: {elapsed_time:.2f}秒")
    if elapsed_time > 0:
        print(f"速度: ~{total_records/elapsed_time:,.0f} 条记录/秒")
    print(f"{'='*60}\n")

    # 查询因子数据统计
    print(f"\n[4/4] 数据库统计:")
    stats = handler.con.execute("""
        SELECT
            COUNT(DISTINCT stock_code) as stock_count,
            MIN(trade_date) as min_date,
            MAX(trade_date) as max_date,
            COUNT(*) as total_records,
            COUNT(DISTINCT factor_id) as factor_count
        FROM factor_data
    """).fetchdf()

    print("\n因子数据统计:")
    print(stats.to_string(index=False))

    # 按日期统计记录数
    date_stats = handler.con.execute("""
        SELECT
            trade_date,
            COUNT(*) as record_count
        FROM factor_data
        GROUP BY trade_date
        ORDER BY trade_date DESC
        LIMIT 10
    """).fetchdf()

    print("\n最近10个交易日的记录数:")
    print(date_stats.to_string(index=False))

    handler.close()


def main():
    parser = argparse.ArgumentParser(description='迁移因子数据到数据库')
    parser.add_argument('--source', required=True, help='源数据目录（parquet文件）')
    parser.add_argument('--db', default='data/stock.db', help='目标数据库路径 (默认: data/stock.db)')
    parser.add_argument('--start-date', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--batch-size', type=int, default=100, help='批量插入大小')

    args = parser.parse_args()

    # 检查数据库文件是否存在
    if not Path(args.db).exists():
        print(f"✗ 错误: 数据库文件不存在: {args.db}")
        print("  请先运行迁移脚本创建数据库:")
        print("  python scripts/migrate_to_duckdb.py --source data/stock/day --target data/stock.db")
        sys.exit(1)

    migrate_factors(
        source_dir=args.source,
        db_path=args.db,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()

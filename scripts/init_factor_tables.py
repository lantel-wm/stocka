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


# Alpha158 因子定义（与 alpha158.py 中的因子列表对应）
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

    for factor_name, category, desc in ALPHA158_FACTORS:
        try:
            factor_id = handler.register_factor(factor_name, category, desc)
            if factor_id:
                registered_count += 1
                if registered_count <= 10 or registered_count % 20 == 0:
                    print(f"  注册因子: {factor_name} (ID: {factor_id})")
            else:
                skipped_count += 1
        except Exception as e:
            print(f"  ✗ 注册因子 {factor_name} 失败: {e}")

    print(f"\n✓ 成功注册 {registered_count} 个因子定义")
    if skipped_count > 0:
        print(f"  (跳过 {skipped_count} 个已存在的因子)")

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

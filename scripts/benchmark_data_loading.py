"""
数据加载性能对比测试
对比CSV vs Parquet的加载速度
"""

import time
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quant_framework.data.data_handler import DataHandler

hs300_codes = ['000001', '000002', '000063', '000100', '000157', '000166', '000301', '000333', '000338', '000408', '000425', '000538', '000568', '000596', '000617', '000625', '000630', '000651', '000661', '000708', '000725', '000768', '000776', '000786', '000792', '000807', '000858', '000876', '000895', '000938', '000963', '000975', '000977', '000983', '000999', '001391', '001965', '001979', '002001', '002027', '002028', '002049', '002050', '002074', '002142', '002179', '002230', '002236', '002241', '002252', '002304', '002311', '002352', '002371', '002384', '002415', '002422', '002459', '002460', '002463', '002466', '002475', '002493', '002594', '002600', '002601', '002625', '002648', '002709', '002714', '002736', '002916', '002920', '002938', '003816', '300014', '300015', '300033', '300059', '300122', '300124', '300251', '300274', '300308', '300316', '300347', '300394', '300408', '300413', '300418', '300433', '300442', '300476', '300498', '300502', '300628', '300661', '300750', '300759', '300760', '300782', '300803', '300832', '300866', '300896', '300979', '300999', '301236', '301269', '302132', '600000', '600009', '600010', '600011', '600015', '600016', '600018', '600019', '600023', '600025', '600026', '600027', '600028', '600029', '600030', '600031', '600036', '600039', '600048', '600050', '600061', '600066', '600085', '600089', '600104', '600111', '600115', '600150', '600160', '600161', '600176', '600183', '600188', '600196', '600219', '600233', '600276', '600309', '600346', '600362', '600372', '600377', '600406', '600415', '600426', '600436', '600438', '600460', '600482', '600489', '600515', '600519', '600522', '600547', '600570', '600584', '600585', '600588', '600600', '600660', '600674', '600690', '600741', '600760', '600795', '600803', '600809', '600845', '600875', '600886', '600887', '600893', '600900', '600905', '600918', '600919', '600926', '600930', '600938', '600941', '600958', '600989', '600999', '601006', '601009', '601012', '601018', '601021', '601058', '601059', '601066', '601077', '601088', '601100', '601111', '601117', '601127', '601136', '601138', '601166', '601169', '601186', '601211', '601225', '601229', '601236', '601238', '601288', '601298', '601318', '601319', '601328', '601336', '601360', '601377', '601390', '601398', '601456', '601600', '601601', '601607', '601618', '601628', '601633', '601658', '601668', '601669', '601688', '601689', '601698', '601728', '601766', '601788', '601800', '601808', '601816', '601818', '601825', '601838', '601857', '601868', '601872', '601877', '601878', '601881', '601888', '601898', '601899', '601901', '601916', '601919', '601939', '601985', '601988', '601995', '601998', '603019', '603195', '603259', '603260', '603288', '603296', '603369', '603392', '603501', '603799', '603893', '603986', '603993', '605117', '605499', '688008', '688009', '688012', '688036', '688041', '688047', '688082', '688111', '688126', '688169', '688187', '688223', '688256', '688271', '688303', '688396', '688472', '688506', '688981']

def validate_data_consistency(csv_handler, parquet_handler, sample_size=1000):
    """
    验证CSV和Parquet加载的数据是否一致

    Args:
        csv_handler: CSV格式加载的handler
        parquet_handler: Parquet格式加载的handler
        sample_size: 验证的样本数量

    Returns:
        dict: 验证结果
    """
    print(f"\n{'='*60}")
    print("数据一致性验证")
    print(f"{'='*60}")

    results = {
        'passed': True,
        'errors': [],
        'warnings': []
    }

    # 获取所有数据
    csv_data = csv_handler.get_all_data()
    parquet_data = parquet_handler.get_all_data()

    # 1. 验证数据形状
    print("\n1. 验证数据形状...")
    if csv_data.shape == parquet_data.shape:
        print(f"   ✓ 数据形状相同: {csv_data.shape}")
    else:
        results['passed'] = False
        results['errors'].append(f"数据形状不同: CSV={csv_data.shape}, Parquet={parquet_data.shape}")
        print(f"   ✗ 数据形状不同: CSV={csv_data.shape}, Parquet={parquet_data.shape}")

    # 2. 验证列名
    print("\n2. 验证列名...")
    csv_cols = set(csv_data.columns)
    parquet_cols = set(parquet_data.columns)

    if csv_cols == parquet_cols:
        print(f"   ✓ 列名相同: {len(csv_cols)} 列")
    else:
        missing_in_parquet = csv_cols - parquet_cols
        extra_in_parquet = parquet_cols - csv_cols
        if missing_in_parquet:
            results['warnings'].append(f"Parquet缺少列: {missing_in_parquet}")
            print(f"   ⚠ Parquet缺少列: {missing_in_parquet}")
        if extra_in_parquet:
            results['warnings'].append(f"Parquet多出列: {extra_in_parquet}")
            print(f"   ⚠ Parquet多出列: {extra_in_parquet}")

    # 3. 验证股票代码
    print("\n3. 验证股票代码...")
    csv_codes = set(csv_handler.get_all_codes())
    parquet_codes = set(parquet_handler.get_all_codes())

    if csv_codes == parquet_codes:
        print(f"   ✓ 股票代码相同: {len(csv_codes)} 只股票")
    else:
        missing = csv_codes - parquet_codes
        extra = parquet_codes - csv_codes
        if missing:
            results['errors'].append(f"Parquet缺少股票: {len(missing)} 只")
            print(f"   ✗ Parquet缺少 {len(missing)} 只股票")
        if extra:
            results['warnings'].append(f"Parquet多出股票: {len(extra)} 只")
            print(f"   ⚠ Parquet多出 {len(extra)} 只股票")

    # 4. 验证日期范围
    print("\n4. 验证日期范围...")
    csv_dates = csv_handler.get_available_dates()
    parquet_dates = parquet_handler.get_available_dates()

    if csv_dates == parquet_dates:
        print(f"   ✓ 日期范围相同: {len(csv_dates)} 个交易日")
        print(f"     {csv_dates[0]} 至 {csv_dates[-1]}")
    else:
        results['errors'].append(f"日期范围不同")
        print(f"   ✗ 日期范围不同")
        print(f"     CSV: {len(csv_dates)} 天 ({csv_dates[0]} 至 {csv_dates[-1]})")
        print(f"     Parquet: {len(parquet_dates)} 天 ({parquet_dates[0]} 至 {parquet_dates[-1]})")

    # 5. 随机抽样验证数值
    print(f"\n5. 验证数据值（随机抽样 {sample_size} 条）...")

    # 重置索引以便采样
    csv_reset = csv_data.reset_index()
    parquet_reset = parquet_data.reset_index()

    # 随机选择样本
    if len(csv_reset) > sample_size:
        sample_indices = csv_reset.sample(n=sample_size, random_state=42).index
    else:
        sample_indices = csv_reset.index

    # 只检查数值列
    numeric_cols = csv_reset.select_dtypes(include=[np.number]).columns.tolist()

    all_match = True
    max_diff = 0
    diff_count = 0

    for idx in sample_indices:
        csv_row = csv_reset.loc[idx]
        parquet_row = parquet_reset.loc[idx]

        for col in numeric_cols:
            csv_val = csv_row[col]
            parquet_val = parquet_row[col]

            # 处理NaN
            if pd.isna(csv_val) and pd.isna(parquet_val):
                continue
            elif pd.isna(csv_val) or pd.isna(parquet_val):
                all_match = False
                diff_count += 1
                if diff_count <= 3:  # 只显示前3个错误
                    print(f"   ✗ 行 {idx}, 列 {col}: CSV={csv_val}, Parquet={parquet_val}")
                continue

            # 比较数值（考虑浮点数精度）
            try:
                diff = abs(float(csv_val) - float(parquet_val))
                if diff > 1e-6:  # 允许微小的浮点数误差
                    all_match = False
                    diff_count += 1
                    max_diff = max(max_diff, diff)
                    if diff_count <= 3:
                        print(f"   ✗ 行 {idx}, 列 {col}: CSV={csv_val}, Parquet={parquet_val}, 差异={diff}")
            except (TypeError, ValueError):
                # 非数值类型，直接比较
                if csv_val != parquet_val:
                    all_match = False
                    diff_count += 1
                    if diff_count <= 3:
                        print(f"   ✗ 行 {idx}, 列 {col}: CSV={csv_val}, Parquet={parquet_val}")

    if all_match:
        print(f"   ✓ 所有样本数据值匹配")
    else:
        print(f"   ⚠ 发现 {diff_count} 处数据差异")
        if max_diff > 0:
            print(f"   ⚠ 最大差异: {max_diff}")
        if diff_count > sample_size * 0.01:  # 超过1%的差异
            results['passed'] = False
            results['errors'].append(f"数据值差异过多: {diff_count}/{sample_size}")
        else:
            results['warnings'].append(f"少量数据值差异: {diff_count}/{sample_size} (可能是浮点精度)")

    # 6. 验证数据类型
    print("\n6. 验证数据类型...")
    csv_dtypes = csv_data.dtypes
    parquet_dtypes = parquet_data.dtypes

    dtype_diffs = []
    for col in csv_data.columns:
        if col in parquet_data.columns:
            if str(csv_dtypes[col]) != str(parquet_dtypes[col]):
                dtype_diffs.append(f"{col}: {csv_dtypes[col]} vs {parquet_dtypes[col]}")

    if not dtype_diffs:
        print(f"   ✓ 数据类型一致")
    else:
        results['warnings'].append(f"数据类型差异: {len(dtype_diffs)} 列")
        print(f"   ⚠ {len(dtype_diffs)} 列数据类型不同:")
        for diff in dtype_diffs[:3]:  # 只显示前3个
            print(f"     - {diff}")

    # 总结
    print(f"\n{'='*60}")
    print("验证结果总结")
    print(f"{'='*60}")

    if results['passed']:
        print("✓ 数据验证通过！CSV 和 Parquet 数据一致")
    else:
        print("✗ 数据验证失败！发现重大差异")

    if results['warnings']:
        print(f"\n⚠ 发现 {len(results['warnings'])} 个警告:")
        for warning in results['warnings']:
            print(f"  - {warning}")

    if results['errors']:
        print(f"\n✗ 发现 {len(results['errors'])} 个错误:")
        for error in results['errors']:
            print(f"  - {error}")

    return results


def benchmark_loading(data_path, use_parquet=True, num_workers=4):
    """测试数据加载性能"""

    print(f"\n{'='*60}")
    format_name = "Parquet" if use_parquet else "CSV"
    print(f"测试 {format_name} 格式加载性能")
    print(f"{'='*60}")

    start_time = time.time()

    # 创建统一的 DataHandler
    handler = DataHandler(
        data_path=data_path,
        use_parquet=use_parquet,
        num_workers=num_workers if use_parquet else 1,  # CSV使用单进程
        # stock_whitelist=hs300_codes
    )

    handler.load_data()

    elapsed_time = time.time() - start_time

    info = handler.get_data_info()

    print(f"\n{format_name} 格式加载结果:")
    print(f"  加载时间: {elapsed_time:.2f} 秒")
    print(f"  股票数量: {info['stock_count']}")
    print(f"  总数据点: {info['total_data_points']:,}")
    print(f"  吞吐量: {info['total_data_points']/elapsed_time:,.0f} 行/秒")

    return elapsed_time, info, handler


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='数据加载性能对比')
    parser.add_argument('--csv-dir', type=str,
                       default='data/factor/day/alpha158',
                       help='CSV文件目录')
    parser.add_argument('--parquet-dir', type=str,
                       default='data/factor/day/alpha158_parquet',
                       help='Parquet文件目录')
    parser.add_argument('--workers', type=int, default=4,
                       help='并行进程数')
    parser.add_argument('--validate', action='store_true',
                       help='是否执行数据一致性验证（需要两种格式都存在）')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='数据验证的样本数量')

    args = parser.parse_args()
    
    # 测试Parquet加载
    parquet_time = parquet_info = parquet_handler = None
    if os.path.exists(args.parquet_dir):
        parquet_time, parquet_info, parquet_handler = benchmark_loading(
            args.parquet_dir,
            use_parquet=True,
            num_workers=args.workers
        )

    # 测试CSV加载
    csv_time = csv_info = csv_handler = None
    if os.path.exists(args.csv_dir):
        csv_time, csv_info, csv_handler = benchmark_loading(
            args.csv_dir,
            use_parquet=False
        )

    # 性能对比
    print(f"\n{'='*60}")
    print("性能对比总结")
    print(f"{'='*60}")

    if csv_time and parquet_time:
        speedup = csv_time / parquet_time
        print(f"CSV加载时间:    {csv_time:.2f} 秒")
        print(f"Parquet加载时间: {parquet_time:.2f} 秒")
        print(f"速度提升:       {speedup:.1f}x")
        print(f"时间节省:       {((1 - 1/speedup) * 100):.1f}%")
    elif parquet_time:
        print(f"Parquet加载时间: {parquet_time:.2f} 秒")
        print("\n提示: CSV目录不存在，无法对比")
        print(f"如需对比，请确保CSV目录存在: {args.csv_dir}")
    else:
        print("\n提示: Parquet目录不存在")
        print(f"请先运行: python scripts/convert_to_parquet.py --data-dir {args.csv_dir}")

    # 数据一致性验证
    if args.validate and csv_handler and parquet_handler:
        validation_result = validate_data_consistency(
            csv_handler,
            parquet_handler,
            sample_size=args.sample_size
        )

        # 根据验证结果设置退出码
        if not validation_result['passed']:
            sys.exit(1)
    elif args.validate:
        print("\n提示: 需要CSV和Parquet两种格式都存在才能进行数据验证")
        print("请确保两个目录都存在并包含数据")

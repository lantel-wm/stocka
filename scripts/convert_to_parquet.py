"""
将CSV文件转换为Parquet格式以加速读取
Parquet格式优势：
1. 列式存储，只读取需要的列
2. 高压缩率，节省磁盘空间
3. 读取速度比CSV快5-10倍
"""

import os
import sys
import glob
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 添加项目根目录到 Python 路径（如果需要）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from quant_framework.utils.logger import get_logger

logger = get_logger(__name__)


def convert_single_csv(args):
    """转换单个CSV文件为Parquet"""
    csv_path, output_dir = args
    try:
        # 读取CSV
        df = pd.read_csv(csv_path)

        # 获取股票代码
        code = os.path.basename(csv_path).replace('.csv', '')

        # 添加code列（如果需要）
        if 'code' not in df.columns:
            df['code'] = code

        # 设置parquet输出路径
        parquet_path = os.path.join(output_dir, f"{code}.parquet")

        # 保存为parquet格式
        df.to_parquet(parquet_path, index=False, compression='snappy')

        return True
    except Exception as e:
        logger.error(f"转换失败 {csv_path}: {e}")
        return False


def convert_csv_to_parquet(data_dir, output_dir=None, num_workers=None):
    """
    批量转换CSV文件为Parquet格式

    Args:
        data_dir: CSV文件所在目录
        output_dir: Parquet输出目录（默认为data_dir_parquet）
        num_workers: 并行进程数（默认为CPU核心数）
    """
    if output_dir is None:
        output_dir = data_dir + '_parquet'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        logger.error(f"未找到CSV文件: {data_dir}")
        return

    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    logger.info(f"输出目录: {output_dir}")

    # 设置并行进程数
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # 保留一个核心

    # 准备参数
    args_list = [(csv_path, output_dir) for csv_path in csv_files]

    # 使用多进程并行转换
    logger.info(f"使用 {num_workers} 个进程进行转换...")
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_single_csv, args_list),
            total=len(args_list),
            desc="转换进度"
        ))

    success_count = sum(results)
    logger.info(f"\n转换完成！")
    logger.info(f"成功: {success_count}/{len(csv_files)}")
    logger.info(f"失败: {len(csv_files) - success_count}/{len(csv_files)}")

    # 比较文件大小
    original_size = sum(os.path.getsize(f) for f in csv_files) / (1024**3)
    parquet_size = sum(
        os.path.getsize(os.path.join(output_dir, f.replace('.csv', '.parquet')))
        for f in csv_files
        if os.path.exists(os.path.join(output_dir, f.replace('.csv', '.parquet')))
    ) / (1024**3)

    logger.info(f"\n存储空间对比:")
    logger.info(f"  CSV总大小:  {original_size:.2f} GB")
    logger.info(f"  Parquet总大小: {parquet_size:.2f} GB")
    logger.info(f"  压缩率: {((1 - parquet_size/original_size) * 100):.1f}%")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='将CSV文件转换为Parquet格式')
    parser.add_argument('--data-dir', type=str, default='data/factor/day/alpha158',
                       help='CSV文件所在目录')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Parquet输出目录（可选）')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行进程数（默认为CPU核心数-1）')

    args = parser.parse_args()

    convert_csv_to_parquet(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_workers=args.workers
    )

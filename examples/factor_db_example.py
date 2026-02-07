#!/usr/bin/env python3
"""
因子数据库使用示例

演示如何使用 DataHandler 的因子功能进行因子存储和查询
"""

from datetime import date
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant_framework.data.data_handler import DataHandler


def example_factor_init():
    """示例1: 初始化因子表"""
    print("\n" + "="*60)
    print("示例1: 初始化因子表")
    print("="*60)

    handler = DataHandler('data/stock.db')

    # 初始化因子表（创建表结构和索引）
    handler._init_factor_tables()
    print("✓ 因子表初始化成功")

    # 查看表信息
    tables = handler.con.execute("SHOW TABLES").fetchdf()
    print("\n数据库表列表:")
    print(tables.to_string(index=False))

    handler.close()


def example_register_factors():
    """示例2: 注册因子定义"""
    print("\n" + "="*60)
    print("示例2: 注册因子定义")
    print("="*60)

    handler = DataHandler('data/stock.db')
    handler._init_factor_tables()

    # 注册 Alpha158 因子（使用预定义的因子ID）
    # 这里使用与 Alpha158.DEFINITIONS 中相同的ID
    alpha158_factors = [
        (18, 'MA5', '5日均线'),
        (19, 'MA10', '10日均线'),
        (20, 'MA20', '20日均线'),
        (63, 'RSV5', '5日RSV'),
        (64, 'RSV10', '10日RSV'),
    ]

    for factor_id, factor_name, desc in alpha158_factors:
        try:
            result_id = handler.register_factor(
                factor_id,
                factor_name,
                'alpha158',
                desc,
                on_conflict='skip'  # 跳过已存在的因子
            )
            print(f"  注册因子: {factor_name} (ID: {result_id})")
        except Exception as e:
            print(f"  ✗ 注册因子 {factor_name} 失败: {e}")

    # 查看所有注册的因子
    all_factors = handler.get_available_factors()
    print(f"\n总共注册了 {len(all_factors)} 个因子")

    handler.close()


def example_save_factors():
    """示例3: 保存因子值"""
    print("\n" + "="*60)
    print("示例3: 保存因子值")
    print("="*60)

    handler = DataHandler('data/stock.db')
    handler._init_factor_tables()

    # 注册因子（指定ID）
    handler.register_factor(18, 'MA5', 'alpha158', '5日均线', on_conflict='skip')
    handler.register_factor(19, 'MA10', 'alpha158', '10日均线', on_conflict='skip')

    # 创建测试数据（3只股票，2个因子）
    factor_df = pd.DataFrame({
        'MA5': [1.02, 0.98, 1.01, 0.99, 1.03],
        'MA10': [0.99, 1.02, 0.97, 1.01, 0.98]
    }, index=['000001', '000002', '600000', '600519', '000858'])

    print(f"\n保存 {len(factor_df)} 只股票的因子数据...")
    print("数据示例:")
    print(factor_df.head())

    # 保存到数据库
    handler.save_factors(factor_df, date(2024, 1, 10))
    print("✓ 因子数据保存成功")

    handler.close()


def example_query_cross_section():
    """示例4: 查询截面因子数据"""
    print("\n" + "="*60)
    print("示例4: 查询截面因子数据")
    print("="*60)

    handler = DataHandler('data/stock.db')

    # 查询某日所有股票的因子
    cross_section = handler.get_factor_cross_section(
        date(2024, 1, 10),
        factor_names=['MA5', 'MA10']
    )

    if not cross_section.empty:
        print(f"\n查询到 {len(cross_section)} 只股票的因子数据")
        print("\n截面数据（前5只股票）:")
        print(cross_section.head())
    else:
        print("\n未查询到因子数据")

    handler.close()


def example_query_time_series():
    """示例5: 查询时序因子数据"""
    print("\n" + "="*60)
    print("示例5: 查询时序因子数据")
    print("="*60)

    handler = DataHandler('data/stock.db')

    # 查询单只股票的历史因子
    stock_factors = handler.get_stock_factors(
        '000001',
        factor_names=['MA5', 'MA10'],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31)
    )

    if not stock_factors.empty:
        print(f"\n查询到股票 000001 的 {len(stock_factors)} 个交易日的因子数据")
        print("\n时序数据（最近5天）:")
        print(stock_factors.tail())
    else:
        print("\n未查询到因子数据")

    handler.close()


def example_query_wide_format():
    """示例6: 查询宽表格式（用于ML预测）"""
    print("\n" + "="*60)
    print("示例6: 查询宽表格式（用于ML预测）")
    print("="*60)

    handler = DataHandler('data/stock.db')

    # 查询多只股票的多因子（宽表格式）
    wide_df = handler.get_factors_wide(
        trade_date=date(2024, 1, 10),
        stock_codes=['000001', '000002', '600000'],
        factor_names=['MA5', 'MA10']
    )

    if not wide_df.empty:
        print(f"\n查询到 {len(wide_df)} 只股票的因子数据（宽表格式）")
        print("\n宽表数据:")
        print(wide_df)
    else:
        print("\n未查询到因子数据")

    handler.close()


def example_factor_info():
    """示例7: 获取因子定义信息"""
    print("\n" + "="*60)
    print("示例7: 获取因子定义信息")
    print("="*60)

    handler = DataHandler('data/stock.db')

    # 获取因子定义信息
    factor_info = handler.get_factor_info()

    print(f"\n总共定义了 {len(factor_info)} 个因子")
    print("\n因子定义（前10个）:")
    print(factor_info.head(10).to_string(index=False))

    # 按类别统计
    if not factor_info.empty:
        print("\n按类别统计:")
        print(factor_info.groupby('factor_category').size())

    handler.close()


def example_ml_workflow():
    """示例8: ML预测工作流"""
    print("\n" + "="*60)
    print("示例8: ML预测工作流")
    print("="*60)

    handler = DataHandler('data/stock.db')

    # ML模型使用的因子列表（使用预定义的因子ID）
    ml_factors = [
        (18, 'MA5'),
        (19, 'MA10'),
        (20, 'MA20'),
        (63, 'RSV5'),
        (64, 'RSV10'),
    ]

    # 注册因子（指定ID）
    for factor_id, factor_name in ml_factors:
        handler.register_factor(factor_id, factor_name, 'alpha158', f'{factor_name} 因子', on_conflict='skip')

    # 模拟保存计算好的因子
    factor_df = pd.DataFrame({
        'MA5': np.random.randn(100) * 0.1 + 1,
        'MA10': np.random.randn(100) * 0.1 + 1,
        'MA20': np.random.randn(100) * 0.1 + 1,
        'RSV5': np.random.rand(100),
        'RSV10': np.random.rand(100)
    }, index=[f'{i:06d}' for i in range(1, 101)])

    handler.save_factors(factor_df, date(2024, 1, 10))
    print(f"✓ 保存了 {len(factor_df)} 只股票的因子数据")

    # 查询用于ML预测的宽表数据
    prediction_data = handler.get_factors_wide(
        trade_date=date(2024, 1, 10),
        stock_codes=factor_df.index[:10].tolist(),
        factor_names=ml_factors
    )

    print(f"\n准备用于ML预测的数据 ({len(prediction_data)} 只股票, {len(ml_factors)} 个因子)")
    print("\n预测数据示例:")
    print(prediction_data.head())

    handler.close()


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("因子数据库使用示例")
    print("="*60)

    try:
        # 运行示例
        example_factor_init()
        example_register_factors()
        example_save_factors()
        example_query_cross_section()
        example_query_time_series()
        example_query_wide_format()
        example_factor_info()
        example_ml_workflow()

        print("\n" + "="*60)
        print("所有示例运行完成!")
        print("="*60 + "\n")

    except FileNotFoundError as e:
        print(f"\n✗ 错误: {e}")
        print("  请确保已创建 stock.db 数据库")
        print("  运行: python scripts/migrate_to_duckdb.py --source data/stock/day --target data/stock.db")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

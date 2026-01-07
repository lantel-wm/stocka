"""
实盘信号生成示例
展示如何生成每日交易信号
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import date, datetime, timedelta

from quant_framework import (
    DataHandler,
    DoubleMAStrategy,
    RealTimeSignalGenerator
)


def main():
    """主函数"""
    print("=" * 70)
    print("实盘信号生成示例")
    print("=" * 70)
    print()

    # ==================== 第1步：加载数据 ====================
    print("第1步：加载数据...")
    data_handler = DataHandler(
        data_path="data/stock/kline/day",
        min_data_points=100
    )

    try:
        # 加载所有历史数据
        data_handler.load_data()
    except Exception as e:
        print(f"加载数据失败：{e}")
        print("\n请确保数据文件存在于 data/stock/kline/day/ 目录")
        return

    print()

    # ==================== 第2步：创建策略 ====================
    print("第2步：创建策略...")
    strategy = DoubleMAStrategy({
        'short_window': 10,
        'long_window': 30,
        'max_position': 3
    })
    print(f"策略: {strategy.name}")
    print(f"参数: {strategy.params}")
    print()

    # ==================== 第3步：创建信号生成器 ====================
    print("第3步：创建信号生成器...")
    signal_generator = RealTimeSignalGenerator(
        data_handler=data_handler,
        strategy=strategy,
        output_dir="signals"
    )
    print(f"信号输出目录: signals/")
    print()

    # ==================== 第4步：生成今日信号 ====================
    print("第4步：生成今日交易信号...")
    print("-" * 70)

    # 方式1：生成今天的信号
    today = date.today()

    # 检查今天是否为交易日
    available_dates = data_handler.get_available_dates()
    if today not in available_dates:
        print(f"今天 ({today}) 不是交易日")
        print(f"使用最后一个交易日: {available_dates[-1]}")
        target_date = available_dates[-1]
    else:
        target_date = today
        print(f"目标日期: {target_date}")

    print()

    # 生成并导出信号
    report = signal_generator.generate_and_export(
        target_date=target_date,
        output_format="csv"
    )

    # 打印摘要
    print()
    print("信号摘要:")
    print(f"  日期: {report['date']}")
    print(f"  策略: {report['strategy']}")
    print(f"  信号总数: {report['summary']['total_count']}")
    print(f"  - 买入: {report['summary']['buy_count']}")
    print(f"  - 卖出: {report['summary']['sell_count']}")
    print(f"  - 持有: {report['summary']['hold_count']}")
    print()

    # ==================== 第5步：生成历史信号（可选） ====================
    print("第5步：生成历史信号示例...")
    print("-" * 70)

    # 获取最近7个交易日的信号
    end_date = target_date
    start_date = end_date - timedelta(days=30)  # 大约一个月

    print(f"日期范围: {start_date} 至 {end_date}")
    print()

    try:
        history_signals = signal_generator.get_signal_history(
            start_date=start_date,
            end_date=end_date,
            save=True
        )

        print(f"历史信号总数: {len(history_signals)}")
        print("历史信号已保存到 signals/ 目录")

    except Exception as e:
        print(f"生成历史信号时出错：{e}")

    print()
    print("=" * 70)
    print("实盘信号生成完成！")
    print()
    print("提示:")
    print("1. 每日运行此脚本以获取最新的交易信号")
    print("2. 信号文件保存在 signals/ 目录")
    print("3. 可以设置定时任务（如crontab）自动运行")
    print("4. 实际交易前，请务必仔细验证信号的有效性")
    print("=" * 70)


def generate_signals_for_date(target_date_str: str):
    """
    为指定日期生成信号

    Args:
        target_date_str: 目标日期字符串 (YYYY-MM-DD)
    """
    print(f"为 {target_date_str} 生成信号...")

    # 加载数据
    data_handler = DataHandler(
        data_path="data/stock/kline/day",
        min_data_points=100
    )

    try:
        data_handler.load_data()
    except Exception as e:
        print(f"加载数据失败：{e}")
        return

    # 创建策略
    strategy = DoubleMAStrategy({
        'short_window': 10,
        'long_window': 30,
        'max_position': 3
    })

    # 创建信号生成器
    signal_generator = RealTimeSignalGenerator(
        data_handler=data_handler,
        strategy=strategy,
        output_dir="signals"
    )

    # 生成信号
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
    report = signal_generator.generate_and_export(
        target_date=target_date,
        output_format="csv"
    )

    print(f"信号已生成并保存")
    print(f"买入: {report['summary']['buy_count']}")
    print(f"卖出: {report['summary']['sell_count']}")


if __name__ == "__main__":
    # 完整示例
    main()

    # 或者，为特定日期生成信号
    # generate_signals_for_date("2024-01-05")

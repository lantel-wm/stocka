#!/usr/bin/env python
"""
回测命令行工具
使用配置文件运行回测

用法:
    python backtest.py                    # 使用默认配置文件 config.yaml
    python backtest.py --config my.yaml  # 使用指定的配置文件
    python backtest.py -c my.yaml        # 简写形式
"""

import sys
import os
import argparse

# 确保可以导入 quant_framework
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from quant_framework import (
    Config,
    DataHandler,
    BacktestEngine,
    TransactionCost,
    StandardCost,
    calculate_all_metrics,
    Performance
)
from quant_framework.performance.reports import ReportGenerator


def run_backtest(config_path: str):
    """
    根据配置文件运行回测

    Args:
        config_path: 配置文件路径
    """
    print("=" * 70)
    print("A股量化回测框架")
    print("=" * 70)
    print()

    # 加载配置
    print(f"加载配置文件: {config_path}")
    try:
        config = Config(config_path)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        sys.exit(1)

    print("配置文件已加载")
    print()

    # 获取各部分配置
    data_config = config.get_data_config()
    backtest_config = config.get_backtest_config()
    cost_config = config.get_transaction_cost_config()
    risk_config = config.get_risk_control_config()
    performance_config = config.get_performance_config()
    output_config = config.get_output_config()

    # 提前获取回测时间范围（在多处使用）
    backtest_start = backtest_config.get('start_date')
    backtest_end = backtest_config.get('end_date')

    # ==================== 第1步：初始化数据处理器 ====================
    print("=" * 70)
    print("第1步：初始化数据处理器")
    print("=" * 70)

    data_handler = DataHandler(
        data_path=data_config.get('base_path', 'data/stock/kline/day'),
        min_data_points=data_config.get('min_data_points', 100),
        stock_whitelist=data_config.get('stock_whitelist', None)
    )

    try:
        # 从配置文件获取数据加载时间范围
        data_start_date = data_config.get('load_start_date')
        data_end_date = data_config.get('load_end_date')

        # 获取回测时间范围
        backtest_start = backtest_config.get('start_date')
        backtest_end = backtest_config.get('end_date')

        # 验证时间范围
        if data_start_date and data_end_date:
            print(f"数据加载范围: {data_start_date} 至 {data_end_date}")
            print(f"回测范围: {backtest_start} 至 {backtest_end}")

            # 验证回测范围在数据加载范围内
            from datetime import datetime
            data_start = datetime.strptime(data_start_date, '%Y-%m-%d')
            data_end = datetime.strptime(data_end_date, '%Y-%m-%d')
            backtest_start_dt = datetime.strptime(backtest_start, '%Y-%m-%d')
            backtest_end_dt = datetime.strptime(backtest_end, '%Y-%m-%d')

            if backtest_start_dt < data_start:
                print(f"警告: 回测开始日期 ({backtest_start}) 早于数据加载开始日期 ({data_start_date})")
            if backtest_end_dt > data_end:
                print(f"警告: 回测结束日期 ({backtest_end}) 晚于数据加载结束日期 ({data_end_date})")
            print()

        data_handler.load_data(
            start_date=data_start_date,
            end_date=data_end_date
        )
        print(f"数据加载成功")
        print(f"  - 股票数量: {len(data_handler.get_all_codes())}")
        print(f"  - 日期范围: {data_handler.available_dates[0]} 至 {data_handler.available_dates[-1]}")
        print()

    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"加载数据时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ==================== 第2步：创建策略 ====================
    print("=" * 70)
    print("第2步：创建策略")
    print("=" * 70)

    try:
        strategy = config.create_strategy()
        strategy_config = config.get_strategy_config()
        print(f"策略类型: {strategy_config.get('type')}")
        print(f"策略参数: {strategy_config.get('params')}")
        print()

    except Exception as e:
        print(f"创建策略时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ==================== 第3步：运行回测 ====================
    print("=" * 70)
    print("第3步：运行回测")
    print("=" * 70)

    print(f"回测时间范围: {backtest_start} 至 {backtest_end}")
    print(f"初始资金: {backtest_config.get('initial_capital', 1000000):,.0f} 元")
    print()

    # 创建交易成本对象
    # 使用配置文件中的参数创建 TransactionCost
    transaction_cost = TransactionCost(
        commission_rate=cost_config.get('commission_rate', 0.0003),
        stamp_duty_rate=cost_config.get('stamp_duty_rate', 0.001),
        min_commission=cost_config.get('min_commission', 5.0),
        slippage=cost_config.get('slippage', 0.001)
    )

    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        initial_capital=backtest_config.get('initial_capital', 1000000),
        max_single_position_ratio=risk_config.get('max_single_position_ratio', 0.3),
        transaction_cost=transaction_cost
    )

    try:
        results = engine.run(
            start_date=backtest_start,
            end_date=backtest_end,
            verbose=backtest_config.get('verbose', True)
        )
        print()

    except Exception as e:
        print(f"回测运行时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ==================== 第4步：计算绩效指标 ====================
    print("=" * 70)
    print("第4步：计算绩效指标")
    print("=" * 70)

    metrics = calculate_all_metrics(
        portfolio_history=results['portfolio_history'],
        trades=results['trades'],
        initial_capital=results['initial_capital'],
        risk_free_rate=performance_config.get('risk_free_rate', 0.03)
    )

    report = Performance.generate_performance_report(
        metrics=metrics,
        trades=results['trades']
    )
    print("\n" + report)
    print()

    # ==================== 第5步：生成报告 ====================
    print("=" * 70)
    print("第5步：生成报告")
    print("=" * 70)

    try:
        # 创建报告生成器
        report_gen = ReportGenerator(
            output_dir=output_config.get('reports_path', 'reports'),
            create_timestamp_dir=output_config.get('create_timestamp_dir', True)
        )

        # 导出配置
        export_config = output_config.get('export', {})
        if export_config.get('config', True):
            print("保存配置文件...")
            report_gen.save_config_to_report(config)

        # 导出交易记录
        if export_config.get('trades', True) and results['trades']:
            print("导出交易记录...")
            report_gen.export_trades_to_csv(results['trades'])

        # 导出持仓历史
        if export_config.get('portfolio_history', True):
            print("导出持仓历史...")
            report_gen.export_positions_to_csv(results['portfolio_history'])

        # 导出详细持仓
        if export_config.get('detailed_positions', True):
            print("导出详细持仓...")
            report_gen.export_detailed_positions_to_csv(
                portfolio=engine.portfolio,
                data_handler=data_handler,
                start_date=backtest_start,
                end_date=backtest_end
            )

        # 导出绩效指标
        if export_config.get('metrics', True):
            print("导出绩效指标...")
            report_gen.export_metrics_to_json(metrics)

        # 绘制图表
        plots_config = output_config.get('plots', {})
        if plots_config.get('equity_curve', True):
            print("绘制资金曲线...")
            report_gen.plot_equity_curve(
                portfolio_history=results['portfolio_history'],
                save=True,
                show=False
            )

        if plots_config.get('returns_distribution', True):
            print("绘制收益率分布...")
            report_gen.plot_returns_distribution(
                portfolio_history=results['portfolio_history'],
                save=True,
                show=False
            )

        if plots_config.get('drawdown', True):
            print("绘制回撤图...")
            report_gen.plot_drawdown(
                portfolio_history=results['portfolio_history'],
                save=True,
                show=False
            )

        print()
        print("=" * 70)
        print("回测完成！")
        print("=" * 70)

    except Exception as e:
        print(f"生成报告时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='A股量化回测框架 - 基于配置文件运行回测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                      # 使用默认配置文件 config.yaml
  %(prog)s -c my_config.yaml    # 使用指定的配置文件
  %(prog)s --config test.yaml   # 完整参数名
        """
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.yaml',
        metavar='FILE',
        help='配置文件路径（默认: config.yaml）'
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    # 运行回测
    run_backtest(args.config)


if __name__ == "__main__":
    main()

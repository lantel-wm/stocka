"""
简单的回测示例
展示如何使用量化框架进行回测
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant_framework import (
    DataHandler,
    MLStrategy,
    BacktestEngine,
    StandardCost,
    Performance,
    calculate_all_metrics
)


def main():
    """主函数"""
    print("=" * 70)
    print("A股量化回测框架 - 简单回测示例")
    print("=" * 70)
    print()

    # ==================== 第1步：初始化数据处理器 ====================
    print("第1步：加载数据...")
    data_handler = DataHandler(
        data_path="../data/factor/day/alpha158",
        min_data_points=50,
        # stock_whitelist=['000001']  # 只加载平安银行的数据
        use_parquet=True,
        num_workers=32,
    )

    try:
        # 数据加载范围（应该包含回测范围）
        data_load_start = "2024-01-02"  # 数据开始日期
        data_load_end = "2025-12-31"    # 数据结束日期

        # 回测范围
        backtest_start = "2024-01-03"
        backtest_end = "2025-12-31"

        print(f"数据加载范围: {data_load_start} 至 {data_load_end}")
        print(f"回测范围: {backtest_start} 至 {backtest_end}")

        data_handler.load_data(
            start_date=data_load_start,
            end_date=data_load_end
        )
    except FileNotFoundError as e:
        print(f"错误：{e}")
        print("\n请确保数据文件存在于目录")
        return
    except Exception as e:
        print(f"加载数据时出错：{e}")
        return

    print()

    # ==================== 第2步：创建策略 ====================
    print("第2步：创建策略...")
    strategy = MLStrategy({
        'model_path': '../examples/lightgbm_model.pkl',
        'rebalance_days': 7,
        'top_k': 10,
        'stop_loss': 0.03,
    })
    print(f"策略名称: {strategy.name}")
    print(f"策略参数: {strategy.params}")
    print()

    # ==================== 第3步：运行回测 ====================
    print("第3步：运行回测...")
    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        initial_capital=1000000,  # 初始资金100万
        max_single_position_ratio=1.0,  # 单只股票最大仓位比例100%
        transaction_cost=StandardCost()  # 使用标准交易成本
    )

    # 运行回测
    results = engine.run(
        start_date=backtest_start,
        end_date=backtest_end,
        verbose=True
    )

    # ==================== 第4步：绩效分析 ====================
    print("第4步：计算绩效指标...")
    metrics = calculate_all_metrics(
        portfolio_history=results['portfolio_history'],
        trades=results['trades'],
        initial_capital=results['initial_capital'],
        risk_free_rate=0.03
    )

    # 打印详细报告
    report = Performance.generate_performance_report(
        metrics=metrics,
        trades=results['trades']
    )
    print("\n" + report)

    # ==================== 第5步：生成图表和CSV报告（可选）====================
    try:
        print("生成报告和图表...")
        from quant_framework.performance.reports import ReportGenerator

        report_gen = ReportGenerator(output_dir="reports")

        # 导出交易记录到CSV
        print("\n导出交易记录...")
        trades_csv = report_gen.export_trades_to_csv(results['trades'])

        # 导出持仓历史到CSV
        print("导出持仓历史...")
        history_csv = report_gen.export_positions_to_csv(results['portfolio_history'])

        # 导出详细持仓到CSV
        print("导出详细持仓...")
        detailed_csv = report_gen.export_detailed_positions_to_csv(
            portfolio=engine.portfolio,
            data_handler=data_handler,
            start_date=backtest_start,
            end_date=backtest_end
        )

        # 导出绩效指标到JSON（包含交易统计）
        print("导出绩效指标...")
        metrics_json = report_gen.export_metrics_to_json(
            metrics=metrics,
            trade_analysis=results.get('trade_analysis')
        )

        # 导出交易分析详情到CSV
        print("导出交易分析...")
        analysis_csv = report_gen.export_trade_analysis_to_csv(
            trade_analysis=results.get('trade_analysis', {})
        )

        # 绘制资金曲线
        print("绘制资金曲线...")
        report_gen.plot_equity_curve(
            portfolio_history=results['portfolio_history'],
            save=True,
            show=False
        )

        # 绘制收益率分布
        print("绘制收益率分布...")
        report_gen.plot_returns_distribution(
            portfolio_history=results['portfolio_history'],
            save=True,
            show=False
        )

    except Exception as e:
        print(f"生成报告时出错：{e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 70)
    print("回测完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

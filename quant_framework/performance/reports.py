"""
报告生成模块
生成各种格式的回测报告
"""

from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os


class ReportGenerator:
    """
    报告生成器
    生成文本、图表等多种格式的报告
    """

    def __init__(self, output_dir: str = "reports", create_timestamp_dir: bool = True):
        """
        初始化报告生成器

        Args:
            output_dir: 报告输出目录
            create_timestamp_dir: 是否创建带时间戳的子目录（默认True）
        """
        self.base_output_dir = output_dir
        self._ensure_base_output_dir()

        # 创建带时间戳的子目录
        if create_timestamp_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = os.path.join(output_dir, f"backtest_{timestamp}")
        else:
            self.output_dir = output_dir

        self._ensure_output_dir()

    def _ensure_base_output_dir(self) -> None:
        """确保基础输出目录存在"""
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)

    def _ensure_output_dir(self) -> None:
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_text_report(self,
                            metrics: Dict,
                            trades: Optional[List[Dict]] = None,
                            save: bool = True) -> str:
        """
        生成文本报告

        Args:
            metrics: 绩效指标
            trades: 交易记录
            save: 是否保存到文件

        Returns:
            报告文本
        """
        from .analyzer import Performance

        report = Performance.generate_performance_report(metrics, trades)

        if save:
            filename = "backtest_report.txt"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)

            print(f"报告已保存到: {filepath}")

        return report

    def plot_equity_curve(self,
                         portfolio_history: List[Dict],
                         benchmark_history: Optional[pd.DataFrame] = None,
                         save: bool = True,
                         show: bool = False) -> Optional[str]:
        """
        绘制资金曲线

        Args:
            portfolio_history: 每日历史记录
            benchmark_history: benchmark历史数据（可选，DataFrame需包含date和benchmark_value列）
            save: 是否保存图片
            show: 是否显示图片

        Returns:
            图片文件路径（如果保存）
        """
        if not portfolio_history:
            print("没有数据可用于绘图")
            return None

        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'])

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 绘制资金曲线
        ax1.plot(df['date'], df['total_value'], 'b-', linewidth=2, label='Portfolio')
        ax1.axhline(y=df['total_value'].iloc[0],
                   color='gray', linestyle='--',
                   label=f"Initial Capital: {df['total_value'].iloc[0]:,.0f}")

        # 如果有benchmark数据，绘制benchmark曲线
        if benchmark_history is not None and 'benchmark_value' in benchmark_history.columns:
            ax1.plot(benchmark_history['date'], benchmark_history['benchmark_value'],
                    'r-', linewidth=2, label='Benchmark (CSI 300)', alpha=0.7)

        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Assets (CNY)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 绘制回撤
        equity_curve = df['total_value'].values
        cumulative_max = pd.Series(equity_curve).cummax().values
        drawdown = (equity_curve - cumulative_max) / cumulative_max * 100

        ax2.fill_between(df['date'], drawdown, 0, alpha=0.3, color='red')
        ax2.plot(df['date'], drawdown, 'r-', linewidth=1)

        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 格式化x轴日期
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # 保存图片
        filepath = None
        if save:
            filename = "equity_curve.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"资金曲线已保存到: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath

    def plot_returns_distribution(self,
                                  portfolio_history: List[Dict],
                                  save: bool = True,
                                  show: bool = False) -> Optional[str]:
        """
        绘制收益率分布

        Args:
            portfolio_history: 每日历史记录
            save: 是否保存图片
            show: 是否显示图片

        Returns:
            图片文件路径（如果保存）
        """
        if not portfolio_history:
            return None

        df = pd.DataFrame(portfolio_history)
        df['returns'] = df['total_value'].pct_change() * 100
        df = df.dropna(subset=['returns'])

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(df['returns'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(df['returns'].mean(), color='red',
                  linestyle='--', linewidth=2,
                  label=f"Mean: {df['returns'].mean():.2f}%")

        ax.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Daily Returns (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filepath = None
        if save:
            filename = "returns_distribution.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"收益率分布图已保存到: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath

    def plot_drawdown(self,
                     portfolio_history: List[Dict],
                     save: bool = True,
                     show: bool = False) -> Optional[str]:
        """
        绘制回撤图

        Args:
            portfolio_history: 每日历史记录
            save: 是否保存图片
            show: 是否显示图片

        Returns:
            图片文件路径（如果保存）
        """
        if not portfolio_history:
            print("没有数据可用于绘图")
            return None

        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'])

        # 计算回撤
        df['cummax'] = df['total_value'].cummax()
        df['drawdown'] = (df['total_value'] - df['cummax']) / df['cummax']

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制回撤曲线
        ax.fill_between(df['date'], df['drawdown'] * 100, 0,
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(df['date'], df['drawdown'] * 100, color='red', linewidth=1)

        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)

        plt.tight_layout()

        filepath = None
        if save:
            filename = "drawdown.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"回撤图已保存到: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath

    def export_trades_to_csv(self,
                            trades: List[Dict],
                            filename: Optional[str] = None) -> str:
        """
        导出交易记录到CSV文件

        Args:
            trades: 交易记录列表
            filename: 文件名（可选）

        Returns:
            CSV文件路径
        """
        if not trades:
            print("没有交易记录")
            return ""

        # 转换为DataFrame
        df = pd.DataFrame(trades)

        # 重命名列为英文
        column_mapping = {
            'date': 'Date',
            'code': 'Stock Code',
            'action': 'Action',
            'shares': 'Shares',
            'price': 'Price',
            'amount': 'Amount',
            'commission': 'Commission',
            'reason': 'Reason'
        }
        df = df.rename(columns=column_mapping)

        # 选择需要的列
        columns_order = ['Date', 'Stock Code', 'Action', 'Shares',
                        'Price', 'Amount', 'Commission', 'Reason']
        df = df[[col for col in columns_order if col in df.columns]]

        # 生成文件名
        if not filename:
            filename = "trades.csv"

        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        print(f"交易记录已导出到: {filepath}")
        print(f"  - 总交易次数: {len(trades)}")
        print(f"  - 买入: {len([t for t in trades if t.get('action') == 'buy'])}")
        print(f"  - 卖出: {len([t for t in trades if t.get('action') == 'sell'])}")

        return filepath

    def export_positions_to_csv(self,
                               portfolio_history: List[Dict],
                               filename: Optional[str] = None) -> str:
        """
        导出持仓历史到CSV文件

        Args:
            portfolio_history: 每日历史记录
            filename: 文件名（可选）

        Returns:
            CSV文件路径
        """
        if not portfolio_history:
            print("没有持仓历史记录")
            return ""

        # 转换为DataFrame
        df = pd.DataFrame(portfolio_history)

        # 重命名列为英文
        column_mapping = {
            'date': 'Date',
            'cash': 'Cash',
            'market_value': 'Market Value',
            'total_value': 'Total Value',
            'positions_count': 'Positions Count'
        }
        df = df.rename(columns=column_mapping)

        # 选择需要的列
        columns_order = ['Date', 'Cash', 'Market Value', 'Total Value', 'Positions Count']
        df = df[[col for col in columns_order if col in df.columns]]

        # 生成文件名
        if not filename:
            filename = "portfolio_history.csv"

        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        print(f"持仓历史已导出到: {filepath}")
        print(f"  - 记录天数: {len(df)}")
        print(f"  - 起始日期: {df['Date'].iloc[0]}")
        print(f"  - 结束日期: {df['Date'].iloc[-1]}")

        return filepath

    def export_detailed_positions_to_csv(self,
                                        portfolio,
                                        data_handler,
                                        start_date,
                                        end_date,
                                        filename: Optional[str] = None) -> str:
        """
        导出详细持仓信息到CSV文件（包含每只股票的持仓）

        Args:
            portfolio: 投资组合对象
            data_handler: 数据处理器
            start_date: 开始日期
            end_date: 结束日期
            filename: 文件名（可选）

        Returns:
            CSV文件路径
        """
        # 获取每日历史
        history = portfolio.get_daily_history()

        # 构建详细持仓记录
        detailed_records = []

        for daily_record in history:
            date = daily_record['date']

            # 获取当日所有持仓
            positions = portfolio.get_all_positions()

            for code, position in positions.items():
                if position.shares > 0:
                    detailed_records.append({
                        'Date': date.strftime('%Y-%m-%d'),
                        'Stock Code': code,
                        'Shares': position.shares,
                        'Avg Price': position.avg_price,
                        'Current Price': position.current_price,
                        'Market Value': position.market_value,
                        'Cost Value': position.cost_value,
                        'Profit/Loss': position.profit_loss,
                        'Profit/Loss %': position.profit_loss_ratio * 100
                    })

        if not detailed_records:
            print("没有详细持仓记录")
            return ""

        # 转换为DataFrame
        df = pd.DataFrame(detailed_records)

        # 生成文件名
        if not filename:
            filename = "detailed_positions.csv"

        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        print(f"详细持仓已导出到: {filepath}")
        print(f"  - 总持仓记录: {len(detailed_records)}")

        return filepath

    def export_all_to_csv(self,
                         trades: List[Dict],
                         portfolio_history: List[Dict],
                         portfolio=None,
                         data_handler=None,
                         start_date=None,
                         end_date=None) -> Dict[str, str]:
        """
        导出所有数据到CSV文件

        Args:
            trades: 交易记录
            portfolio_history: 持仓历史
            portfolio: 投资组合对象（可选，用于详细持仓）
            data_handler: 数据处理器（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            文件路径字典
        """
        filepaths = {}

        # 导出交易记录
        if trades:
            filepaths['trades'] = self.export_trades_to_csv(trades)

        # 导出持仓历史
        if portfolio_history:
            filepaths['portfolio_history'] = self.export_positions_to_csv(portfolio_history)

        # 导出详细持仓（如果提供portfolio）
        if portfolio and data_handler and start_date and end_date:
            filepaths['detailed_positions'] = self.export_detailed_positions_to_csv(
                portfolio, data_handler, start_date, end_date
            )

        return filepaths

    def export_metrics_to_json(self,
                                metrics: Dict,
                                trade_analysis: Optional[Dict] = None,
                                filename: Optional[str] = None) -> str:
        """
        导出绩效指标到JSON文件

        Args:
            metrics: 绩效指标字典
            trade_analysis: 交易分析结果（可选，用于提取win_rate）
            filename: 文件名（可选）

        Returns:
            JSON文件路径
        """
        import json

        # 生成文件名
        if not filename:
            filename = "metrics.json"

        filepath = os.path.join(self.output_dir, filename)

        # 将metrics转换为可序列化的格式
        serializable_metrics = {}
        for key, value in metrics.items():
            # 处理numpy和pandas类型
            if hasattr(value, 'item'):  # numpy类型
                serializable_metrics[key] = float(value.item())
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_metrics[key] = value
            else:
                # 其他类型转换为字符串
                serializable_metrics[key] = str(value)

        # 从trade_analysis中提取win_rate添加到metrics
        if trade_analysis and 'win_rate' in trade_analysis:
            serializable_metrics['win_rate'] = trade_analysis['win_rate']

        # 保存到JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

        print(f"绩效指标已导出到: {filepath}")
        print(f"  - 指标数量: {len(serializable_metrics)}")
        if trade_analysis:
            print(f"  - 完整交易: {trade_analysis.get('completed_trades', 0)} 笔")
            print(f"  - 胜率: {trade_analysis.get('win_rate', 0):.2f}%")

        return filepath

    def export_trade_analysis_to_csv(self,
                                     trade_analysis: Dict,
                                     filename: Optional[str] = None) -> str:
        """
        导出交易分析详情到CSV文件

        Args:
            trade_analysis: 交易分析结果字典
            filename: 文件名（可选）

        Returns:
            CSV文件路径
        """
        if not trade_analysis or not trade_analysis.get('trade_details'):
            print("没有交易详情记录")
            return ""

        # 获取交易详情
        trade_details = trade_analysis['trade_details']

        # 转换为DataFrame
        df = pd.DataFrame(trade_details)

        # 重命名列为英文
        column_mapping = {
            'code': 'Stock Code',
            'buy_date': 'Buy Date',
            'sell_date': 'Sell Date',
            'buy_price': 'Buy Price',
            'sell_price': 'Sell Price',
            'shares': 'Shares',
            'buy_cost': 'Buy Cost',
            'sell_income': 'Sell Income',
            'pnl': 'Profit/Loss',
            'pnl_pct': 'Profit/Loss %',
            'buy_reason': 'Buy Reason',
            'sell_reason': 'Sell Reason'
        }
        df = df.rename(columns=column_mapping)

        # 选择需要的列
        columns_order = ['Stock Code', 'Buy Date', 'Sell Date', 'Buy Price', 'Sell Price',
                        'Shares', 'Buy Cost', 'Sell Income', 'Profit/Loss', 'Profit/Loss %',
                        'Buy Reason', 'Sell Reason']
        df = df[[col for col in columns_order if col in df.columns]]

        # 生成文件名
        if not filename:
            filename = "trade_analysis.csv"

        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        # 打印统计信息
        print(f"交易分析已导出到: {filepath}")
        print(f"  - 完整交易: {trade_analysis.get('completed_trades', 0)} 笔")
        print(f"  - 盈利交易: {trade_analysis.get('winning_trades', 0)} 笔")
        print(f"  - 亏损交易: {trade_analysis.get('losing_trades', 0)} 笔")
        print(f"  - 胜率: {trade_analysis.get('win_rate', 0):.2f}%")
        print(f"  - 总盈利: {trade_analysis.get('total_profit', 0):,.2f} 元")
        print(f"  - 总亏损: {trade_analysis.get('total_loss', 0):,.2f} 元")
        print(f"  - 净盈亏: {trade_analysis.get('total_pnl', 0):,.2f} 元")

        return filepath

    def save_config_to_report(self,
                               config_obj,
                               filename: Optional[str] = None) -> str:
        """
        保存配置文件到报告目录

        Args:
            config_obj: Config对象或配置字典
            filename: 文件名（可选）

        Returns:
            配置文件路径
        """
        import yaml

        # 生成文件名
        if not filename:
            filename = "config.yaml"

        filepath = os.path.join(self.output_dir, filename)

        # 获取配置字典
        if hasattr(config_obj, 'to_dict'):
            config_dict = config_obj.to_dict()
        elif isinstance(config_obj, dict):
            config_dict = config_obj
        else:
            raise ValueError("config_obj 必须是 Config 对象或字典")

        # 保存配置文件
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)

        print(f"配置文件已保存到: {filepath}")

        return filepath

    def generate_html_report(self,
                            metrics: Dict,
                            portfolio_history: List[Dict],
                            trades: Optional[List[Dict]] = None,
                            save: bool = True) -> Optional[str]:
        """
        生成HTML报告

        Args:
            metrics: 绩效指标
            portfolio_history: 每日历史记录
            trades: 交易记录
            save: 是否保存

        Returns:
            HTML文件路径（如果保存）
        """
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回测报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
            min-width: 200px;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
        }}
        .metric-value {{
            color: #333;
            font-size: 24px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>量化回测报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>核心指标</h2>
        <div class="metric">
            <div class="metric-label">总收益率</div>
            <div class="metric-value">{metrics.get('total_return', 0)*100:.2f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">年化收益率</div>
            <div class="metric-value">{metrics.get('annual_return', 0)*100:.2f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">最大回撤</div>
            <div class="metric-value">{metrics.get('max_drawdown', 0)*100:.2f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">夏普比率</div>
            <div class="metric-value">{metrics.get('sharpe_ratio', 0):.4f}</div>
        </div>

        <h2>详细指标</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>值</th>
            </tr>
            <tr>
                <td>初始资金</td>
                <td>{metrics.get('initial_capital', 0):,.2f} 元</td>
            </tr>
            <tr>
                <td>最终权益</td>
                <td>{metrics.get('final_value', 0):,.2f} 元</td>
            </tr>
            <tr>
                <td>年化波动率</td>
                <td>{metrics.get('annual_volatility', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>交易日数</td>
                <td>{metrics.get('trading_days', 0)}</td>
            </tr>
        </table>
    </div>
</body>
</html>
"""

        filepath = None
        if save:
            filename = "backtest_report.html"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"HTML报告已保存到: {filepath}")

        return filepath

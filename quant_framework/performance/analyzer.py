"""
绩效分析模块
计算各种回测绩效指标
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import date


class Performance:
    """
    绩效分析类
    计算回测的各种绩效指标
    """

    @staticmethod
    def calculate_metrics(portfolio_history: List[Dict],
                         initial_capital: float = 1000000.0,
                         risk_free_rate: float = 0.03,
                         trading_days_per_year: int = 252) -> Dict:
        """
        计算绩效指标

        Args:
            portfolio_history: 每日历史记录
            initial_capital: 初始资金
            risk_free_rate: 无风险利率（年化）
            trading_days_per_year: 年化交易日数

        Returns:
            绩效指标字典
        """
        if not portfolio_history:
            return {}

        # 转换为DataFrame
        df = pd.DataFrame(portfolio_history)

        # 计算日收益率
        df['returns'] = df['total_value'].pct_change()
        df = df.dropna(subset=['returns'])

        if len(df) == 0:
            return {}

        # 基础指标
        final_value = df['total_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital

        # 年化收益率
        trading_days = len(df)
        years = trading_days / trading_days_per_year
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 波动率
        daily_volatility = df['returns'].std()
        annual_volatility = daily_volatility * np.sqrt(trading_days_per_year)

        # 夏普比率
        daily_rf = risk_free_rate / trading_days_per_year
        excess_returns = df['returns'] - daily_rf
        sharpe_ratio = (excess_returns.mean() / excess_returns.std() *
                       np.sqrt(trading_days_per_year)
                       if excess_returns.std() > 0 else 0)

        # 最大回撤
        equity_curve = df['total_value']
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # 最大回撤持续期
        max_dd_idx = drawdown.idxmin()
        max_dd_date = df.loc[max_dd_idx, 'date']

        # 胜率等指标（需要交易数据）
        win_rate = 0.0
        profit_loss_ratio = 0.0

        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'daily_volatility': daily_volatility,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trading_days': trading_days,
            'years': years,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio
        }

    @staticmethod
    def calculate_trade_metrics(trades: List[Dict]) -> Dict:
        """
        计算交易相关指标

        Args:
            trades: 交易记录列表

        Returns:
            交易指标字典
        """
        if not trades:
            return {
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'win_rate': 0.0,
                'profit_loss_ratio': 0.0
            }

        buy_trades = [t for t in trades if t['action'] == 'buy']
        sell_trades = [t for t in trades if t['action'] == 'sell']

        # 计算每对买卖交易的盈亏
        # 这里简化处理，实际应该配对买卖交易
        profits = []

        return {
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
        }

    @staticmethod
    def calculate_rolling_metrics(equity_curve: pd.Series,
                                  window: int = 20) -> pd.DataFrame:
        """
        计算滚动指标

        Args:
            equity_curve: 资金曲线
            window: 滚动窗口

        Returns:
            包含滚动指标的DataFrame
        """
        df = pd.DataFrame({'equity': equity_curve})
        df['returns'] = df['equity'].pct_change()

        # 滚动收益率
        df['rolling_return'] = df['returns'].rolling(window).sum()

        # 滚动波动率
        df['rolling_volatility'] = df['returns'].rolling(window).std()

        # 滚动夏普比率
        df['rolling_sharpe'] = (df['rolling_return'] / df['rolling_volatility']
                               if df['rolling_volatility'].iloc[-1] > 0 else 0)

        # 滚动最大回撤
        rolling_max = df['equity'].rolling(window).max()
        df['rolling_drawdown'] = (df['equity'] - rolling_max) / rolling_max

        return df

    @staticmethod
    def generate_performance_report(metrics: Dict,
                                   trades: Optional[List[Dict]] = None) -> str:
        """
        生成文本格式的绩效报告

        Args:
            metrics: 绩效指标字典
            trades: 交易记录（可选）

        Returns:
            报告字符串
        """
        if not metrics:
            return "没有可用的绩效指标"

        report = []
        report.append("=" * 60)
        report.append("回测绩效报告")
        report.append("=" * 60)
        report.append("")

        # 基础指标
        report.append("【基础收益指标】")
        report.append(f"初始资金: {metrics['initial_capital']:,.2f} 元")
        report.append(f"最终权益: {metrics['final_value']:,.2f} 元")
        report.append(f"总收益率: {metrics['total_return']*100:.2f}%")
        report.append(f"年化收益率: {metrics['annual_return']*100:.2f}%")
        report.append("")

        # 风险指标
        report.append("【风险指标】")
        report.append(f"日波动率: {metrics['daily_volatility']*100:.4f}%")
        report.append(f"年化波动率: {metrics['annual_volatility']*100:.2f}%")
        report.append(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
        report.append(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
        report.append("")

        # 交易统计
        if trades:
            trade_metrics = Performance.calculate_trade_metrics(trades)
            report.append("【交易统计】")
            report.append(f"总交易次数: {trade_metrics['total_trades']}")
            report.append(f"买入次数: {trade_metrics['buy_trades']}")
            report.append(f"卖出次数: {trade_metrics['sell_trades']}")
            report.append("")

        # 时间统计
        report.append("【时间统计】")
        report.append(f"交易日数: {metrics['trading_days']}")
        report.append(f"回测年数: {metrics['years']:.2f}")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)


def calculate_all_metrics(portfolio_history: List[Dict],
                         trades: List[Dict],
                         initial_capital: float = 1000000.0,
                         risk_free_rate: float = 0.03) -> Dict:
    """
    计算所有绩效指标

    Args:
        portfolio_history: 每日历史记录
        trades: 交易记录
        initial_capital: 初始资金
        risk_free_rate: 无风险利率

    Returns:
        完整的绩效指标字典
    """
    # 计算基础绩效指标
    metrics = Performance.calculate_metrics(
        portfolio_history,
        initial_capital,
        risk_free_rate
    )

    # 计算交易指标
    trade_metrics = Performance.calculate_trade_metrics(trades)

    # 合并结果
    metrics.update(trade_metrics)

    return metrics

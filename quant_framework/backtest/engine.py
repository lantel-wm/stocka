"""
回测引擎模块
协调整个回测流程，确保时间序列正确性和避免未来函数
"""

from datetime import date, datetime
from typing import Optional, Dict, List
import pandas as pd

from ..data.data_handler import DataHandler
from ..strategy.base_strategy import BaseStrategy
from ..portfolio.portfolio import Portfolio
from ..execution.transaction_cost import TransactionCost


class BacktestEngine:
    """
    回测引擎
    协调数据、策略、投资组合和交易执行，完成完整的回测流程
    """

    def __init__(self,
                 data_handler: DataHandler,
                 strategy: BaseStrategy,
                 initial_capital: float = 1000000.0,
                 max_single_position_ratio: float = 0.3,
                 transaction_cost: Optional[TransactionCost] = None):
        """
        初始化回测引擎

        Args:
            data_handler: 数据处理器
            strategy: 交易策略
            initial_capital: 初始资金
            transaction_cost: 交易成本模型（可选）
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost or TransactionCost()

        # 初始化投资组合
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            max_single_position_ratio=max_single_position_ratio
        )

        # 回测结果
        self.results = {}
        self.is_running = False
        self.is_completed = False

    def run(self,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            verbose: bool = True) -> Dict:
        """
        运行回测

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            verbose: 是否打印进度信息

        Returns:
            回测结果字典
        """
        self.is_running = True
        start_time = datetime.now()

        if verbose:
            print("=" * 60)
            print("开始回测")
            print("=" * 60)
            print(f"策略: {self.strategy.name}")
            print(f"参数: {self.strategy.params}")
            print(f"初始资金: {self.initial_capital:,.2f} 元")
            print("-" * 60)

        # 获取交易日期列表
        trading_dates = self.data_handler.get_available_dates(
            start_date=str_to_date(start_date) if start_date else None,
            end_date=str_to_date(end_date) if end_date else None
        )

        if not trading_dates:
            raise ValueError("没有可用的交易日期")

        # 确保有足够的数据点用于策略
        min_bars = self.strategy.min_bars
        original_start_date = trading_dates[0]

        if min_bars > 0:
            # 获取所有可用日期
            all_dates = self.data_handler.get_available_dates()

            # 找到原始开始日期在所有数据中的索引
            if original_start_date in all_dates:
                start_index = all_dates.index(original_start_date)

                # 检查是否从开始日期往前有足够的min_bars条数据
                if start_index >= min_bars:
                    # 有足够数据，无需调整
                    if verbose:
                        print(f"策略需要 {min_bars} 条前置bar数据")
                        print(f"数据准备期: {all_dates[start_index - min_bars]} 至 {original_start_date}")
                else:
                    # 数据不足，往前调整开始日期
                    adjusted_start_index = min_bars
                    adjusted_start_date = all_dates[adjusted_start_index]

                    # 重新获取trading_dates，从调整后的日期开始
                    trading_dates = [
                        d for d in trading_dates
                        if d >= adjusted_start_date
                    ]

                    if verbose:
                        print(f"策略需要 {min_bars} 条前置bar数据")
                        print(f"原始开始日期: {original_start_date}（往前不足{min_bars}条数据）")
                        print(f"调整后开始日期: {adjusted_start_date}")
            else:
                # 找不到指定开始日期
                if len(all_dates) > min_bars:
                    trading_dates = all_dates[min_bars:]
                    if verbose:
                        print(f"未找到指定开始日期 {original_start_date}")
                        print(f"调整后开始日期: {trading_dates[0]}（数据准备期: {all_dates[0]} 至 {trading_dates[0]}）")
                else:
                    raise ValueError(f"数据不足：总共只有 {len(all_dates)} 条数据，策略需要 {min_bars} 条")

        if verbose:
            print(f"回测期间: {trading_dates[0]} 至 {trading_dates[-1]}")
            print(f"交易日数: {len(trading_dates)}")
            print("-" * 60)

        # 逐日回测
        for i, current_date in enumerate(trading_dates):
            # 策略生成信号
            signals = self.strategy.on_bar(
                self.data_handler,
                current_date,
                self.portfolio
            )

            # 更新投资组合（执行交易）
            self.portfolio.update(
                current_date,
                signals,
                self.data_handler,
                self.transaction_cost
            )

            # 打印进度
            if verbose and (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(trading_dates)}] "
                      f"{current_date} | "
                      f"总资产: {self.portfolio.total_value:,.2f} | "
                      f"持仓: {self.portfolio.get_position_count()}")

        # 回测完成
        self.is_running = False
        self.is_completed = True

        # 整理结果
        self.results = self._generate_results()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if verbose:
            print("-" * 60)
            print("回测完成")
            print(f"耗时: {duration:.2f} 秒")
            print(f"最终权益: {self.results['final_value']:,.2f} 元")
            print(f"总收益率: {self.results['total_return']*100:.2f}%")
            print(f"交易次数: {len(self.results['trades'])}")
            print("=" * 60)

        return self.results

    def _generate_results(self) -> Dict:
        """生成回测结果"""
        daily_history = self.portfolio.get_daily_history()

        if not daily_history:
            return {}

        # 基础指标
        initial_value = self.initial_capital
        final_value = self.portfolio.total_value
        total_return = (final_value - initial_value) / initial_value

        # 计算日收益率
        df_history = pd.DataFrame(daily_history)
        df_history['returns'] = df_history['total_value'].pct_change()

        # 统计数据
        total_trades = len(self.portfolio.trades)
        buy_trades = len([t for t in self.portfolio.trades if t.action == 'buy'])
        sell_trades = len([t for t in self.portfolio.trades if t.action == 'sell'])

        return {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'portfolio_history': daily_history,
            'trades': [t.to_dict() for t in self.portfolio.trades],
            'strategy_params': self.strategy.params,
            'portfolio': self.portfolio
        }

    def get_results(self) -> Dict:
        """
        获取回测结果

        Returns:
            结果字典
        """
        if not self.is_completed:
            raise ValueError("回测尚未完成，请先调用 run() 方法")

        return self.results

    def reset(self) -> None:
        """重置回测引擎"""
        self.portfolio.reset()
        self.strategy.reset()
        self.results = {}
        self.is_running = False
        self.is_completed = False

    def get_equity_curve(self) -> pd.Series:
        """
        获取资金曲线

        Returns:
            资金曲线Series
        """
        if not self.is_completed:
            raise ValueError("回测尚未完成")

        daily_history = self.portfolio.get_daily_history()
        df = pd.DataFrame(daily_history)

        return pd.Series(
            df['total_value'].values,
            index=pd.to_datetime(df['date'])
        )

    def get_drawdown_series(self) -> pd.Series:
        """
        获取回撤序列

        Returns:
            回撤序列
        """
        equity_curve = self.get_equity_curve()
        cumulative = equity_curve.cummax()
        drawdown = (equity_curve - cumulative) / cumulative

        return drawdown


def str_to_date(date_str: Optional[str]) -> Optional[date]:
    """将字符串转换为日期"""
    if not date_str:
        return None
    return datetime.strptime(date_str, '%Y-%m-%d').date()

"""
实盘信号生成模块
每日生成交易信号并导出
"""

from datetime import date, datetime
from typing import List, Optional
import pandas as pd
import os
import json

from ..data.data_handler import DataHandler
from ..strategy.base_strategy import BaseStrategy, Signal
from ..utils.helpers import ensure_dir


class RealTimeSignalGenerator:
    """
    实盘信号生成器
    基于策略生成当日的交易信号
    """

    def __init__(self,
                 data_handler: DataHandler,
                 strategy: BaseStrategy,
                 output_dir: str = "signals"):
        """
        初始化信号生成器

        Args:
            data_handler: 数据处理器
            strategy: 交易策略
            output_dir: 信号输出目录
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.output_dir = output_dir
        ensure_dir(output_dir)

    def generate_signals(self, target_date: Optional[date] = None) -> List[Signal]:
        """
        生成交易信号

        Args:
            target_date: 目标日期（默认为今天）

        Returns:
            信号列表
        """
        if target_date is None:
            target_date = date.today()

        # 创建一个空的投资组合（用于策略上下文）
        from ..portfolio.portfolio import Portfolio
        dummy_portfolio = Portfolio()

        # 生成信号
        signals = self.strategy.on_bar(
            self.data_handler,
            target_date,
            dummy_portfolio
        )

        return signals

    def generate_daily_report(self, target_date: Optional[date] = None) -> dict:
        """
        生成每日操作报告

        Args:
            target_date: 目标日期

        Returns:
            报告字典
        """
        if target_date is None:
            target_date = date.today()

        signals = self.generate_signals(target_date)

        # 统计信号
        buy_count = len([s for s in signals if s.signal_type == Signal.BUY])
        sell_count = len([s for s in signals if s.signal_type == Signal.SELL])
        hold_count = len([s for s in signals if s.signal_type == Signal.HOLD])

        report = {
            'date': target_date.strftime('%Y-%m-%d'),
            'strategy': self.strategy.name,
            'strategy_params': self.strategy.params,
            'signals': [s.to_dict() for s in signals],
            'summary': {
                'total_count': len(signals),
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count
            }
        }

        return report

    def export_signals(self,
                      signals: List[Signal],
                      output_format: str = "csv",
                      filename: Optional[str] = None) -> str:
        """
        导出信号到文件

        Args:
            signals: 信号列表
            output_format: 输出格式（csv或json）
            filename: 文件名（可选）

        Returns:
            文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if output_format == "csv":
                filename = f"daily_signals_{timestamp}.csv"
            else:
                filename = f"daily_signals_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        if output_format == "csv":
            self._export_csv(signals, filepath)
        elif output_format == "json":
            self._export_json(signals, filepath)
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")

        return filepath

    def _export_csv(self, signals: List[Signal], filepath: str) -> None:
        """导出为CSV格式"""
        data = []
        for signal in signals:
            data.append({
                '日期': signal.date.strftime('%Y-%m-%d') if signal.date else '',
                '股票代码': signal.code,
                '操作类型': signal.signal_type,
                '价格': f"{signal.price:.2f}",
                '仓位权重': f"{signal.weight:.4f}",
                '原因': signal.reason
            })

        df = pd.DataFrame(data)

        if not df.empty:
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"信号已导出到: {filepath}")
        else:
            print("没有信号需要导出")

    def _export_json(self, signals: List[Signal], filepath: str) -> None:
        """导出为JSON格式"""
        data = {
            'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'signals': [s.to_dict() for s in signals]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"信号已导出到: {filepath}")

    def print_signals(self, signals: List[Signal]) -> None:
        """
        打印信号到控制台

        Args:
            signals: 信号列表
        """
        if not signals:
            print("今日无交易信号")
            return

        print("\n" + "=" * 60)
        print(f"交易信号 (日期: {signals[0].date if signals else ''})")
        print("=" * 60)

        for i, signal in enumerate(signals, 1):
            print(f"\n信号 {i}:")
            print(f"  股票代码: {signal.code}")
            print(f"  操作: {signal.signal_type}")
            print(f"  价格: {signal.price:.2f}")
            print(f"  仓位权重: {signal.weight*100:.1f}%")
            print(f"  原因: {signal.reason}")

        print("\n" + "=" * 60)

    def generate_and_export(self,
                           target_date: Optional[date] = None,
                           output_format: str = "csv") -> dict:
        """
        生成信号并导出

        Args:
            target_date: 目标日期
            output_format: 输出格式

        Returns:
            报告字典
        """
        # 生成报告
        report = self.generate_daily_report(target_date)

        # 打印信号
        signals = [Signal() for s in report['signals']]
        for i, s_dict in enumerate(report['signals']):
            if i < len(signals):
                signals[i].date = datetime.strptime(s_dict['date'], '%Y-%m-%d').date()
                signals[i].code = s_dict['code']
                signals[i].signal_type = s_dict['signal_type']
                signals[i].price = s_dict['price']
                signals[i].weight = s_dict['weight']
                signals[i].reason = s_dict['reason']

        self.print_signals(signals)

        # 导出信号
        self.export_signals(signals, output_format)

        return report

    def get_signal_history(self,
                          start_date: date,
                          end_date: date,
                          save: bool = False) -> List[dict]:
        """
        获取历史信号

        Args:
            start_date: 开始日期
            end_date: 结束日期
            save: 是否保存

        Returns:
            历史信号列表
        """
        trading_dates = self.data_handler.get_available_dates(start_date, end_date)
        trading_dates = [d for d in trading_dates if start_date <= d <= end_date]

        all_signals = []

        for trade_date in trading_dates:
            signals = self.generate_signals(trade_date)
            for signal in signals:
                all_signals.append(signal.to_dict())

        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(
                self.output_dir,
                f"signal_history_{start_date}_to_{end_date}_{timestamp}.json"
            )

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'signals': all_signals
                }, f, ensure_ascii=False, indent=2)

            print(f"历史信号已保存到: {filepath}")

        return all_signals

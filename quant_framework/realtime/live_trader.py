"""
实盘交易调度器
协调整个实盘交易流程，包括数据更新、策略执行、信号生成和状态管理
"""

import os
import json
from datetime import date, datetime
from typing import Optional, Dict, List
from pathlib import Path
import pandas as pd

from ..data.data_handler import DataHandler
from ..strategy.base_strategy import BaseStrategy, Signal
from .data_updater import DataUpdater
from ..utils.constraints import TradingCalendar
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LiveTrader:
    """
    实盘交易调度器

    功能：
    - 自动更新数据（调用 DataUpdater）
    - 重新加载数据到 DataHandler
    - 调用策略的 on_bar() 生成交易信号
    - 判断是否为调仓日
    - 导出信号到文件（CSV/JSON）
    - 持久化运行状态
    """

    def __init__(self,
                 strategy: BaseStrategy,
                 data_dir: str,
                 signal_output_dir: str = "signals",
                 log_dir: str = "logs",
                 state_file: str = "live_trader_state.json"):
        """
        初始化实盘交易调度器

        Args:
            strategy: 策略实例
            data_dir: 数据文件目录
            signal_output_dir: 信号输出目录
            log_dir: 日志目录
            state_file: 状态文件路径
        """
        self.strategy = strategy
        self.data_dir = data_dir
        self.signal_output_dir = signal_output_dir
        self.log_dir = log_dir
        self.state_file = state_file

        # 创建必要的目录
        Path(signal_output_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # 初始化数据更新器
        self.data_updater = DataUpdater(
            output_dir=data_dir,
            use_parquet=False,
            delay=0.5
        )

        # 初始化交易日历
        self.trading_calendar = TradingCalendar()

        # 数据处理器（延迟加载）
        self.data_handler: Optional[DataHandler] = None

        # 运行状态
        self.state = {
            'strategy_start_date': None,
            'last_trading_date': None,
            'trading_days_count': 0,
            'last_rebalance_date': None,
            'execution_history': []
        }

        # 尝试加载现有状态
        self._load_state()

        logger.info(f"✓ 实盘交易调度器初始化完成")
        logger.info(f"  - 数据目录: {data_dir}")
        logger.info(f"  - 信号输出目录: {signal_output_dir}")
        logger.info(f"  - 状态文件: {state_file}")

        if self.state['strategy_start_date']:
            logger.info(f"\n已加载历史状态:")
            logger.info(f"  - 策略开始日期: {self.state['strategy_start_date']}")
            logger.info(f"  - 上次交易日期: {self.state['last_trading_date']}")
            logger.info(f"  - 交易日计数: {self.state['trading_days_count']}")
            logger.info(f"  - 上次调仓日: {self.state['last_rebalance_date']}")

    def run(self,
            target_date: date,
            force_rebalance: bool = False,
            export_format: str = "csv") -> Dict:
        """
        运行一次实盘交易流程（不更新数据）

        Args:
            target_date: 目标日期
            force_rebalance: 是否强制调仓（忽略调仓周期）
            export_format: 信号导出格式 ("csv", "json", 或 "both")

        Returns:
            执行结果字典：
            {
                'status': 'success' | 'no_rebalance' | 'error',
                'date': date,
                'message': str,
                'is_rebalance_day': bool,
                'signals_generated': int,
                'signals': List[Signal]
            }
        """
        result = {
            'status': 'error',
            'date': target_date,
            'message': '',
            'is_rebalance_day': False,
            'signals_generated': 0,
            'signals': []
        }

        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"实盘交易运行 - {target_date}")
            logger.info(f"{'='*70}")

            # 1. 加载数据
            if self.data_handler is None:
                logger.info(f"\n[1/5] 加载数据...")
                self.data_handler = DataHandler(
                    data_path=self.data_dir,
                    use_parquet=False
                )
                self.data_handler.load_data()
                logger.info(f"✓ 数据加载完成")
            else:
                logger.info(f"\n[1/5] 使用已加载的数据")

            # 2. 检查目标日期是否为交易日（使用交易日历，不依赖数据文件）
            logger.info(f"\n[2/5] 检查交易日...")

            if not self.trading_calendar.is_trading_day(target_date):
                # 找最近的交易日
                actual_date = self.trading_calendar.get_previous_trading_day(target_date)
                if actual_date is None:
                    result['status'] = 'error'
                    result['message'] = f'没有可用的交易日数据（目标日期：{target_date}）'
                    return result

                logger.warning(f"⚠ {target_date} 不是交易日，使用最近的交易日：{actual_date}")
            else:
                actual_date = target_date
                logger.info(f"✓ {actual_date} 是交易日")

            # 3. 初始化策略实盘模式（首次运行）
            logger.info(f"\n[3/5] 初始化策略...")
            if self.state['strategy_start_date'] is None:
                self.state['strategy_start_date'] = actual_date.strftime('%Y-%m-%d')
                self.strategy.initialize_for_live_trading(
                    data_handler=self.data_handler,
                    strategy_start_date=actual_date
                )
                logger.info(f"✓ 策略首次初始化完成")
            else:
                logger.info(f"✓ 策略已初始化（开始日期：{self.state['strategy_start_date']}）")

            # 4. 判断是否为调仓日
            logger.info(f"\n[4/5] 判断调仓日...")
            is_rebalance_day = self.strategy.is_rebalance_day(actual_date)

            if force_rebalance:
                logger.warning(f"⚠ 强制调仓模式启用")
                is_rebalance_day = True

            result['is_rebalance_day'] = is_rebalance_day

            if is_rebalance_day:
                logger.info(f"✓ {actual_date} 是调仓日")
            else:
                logger.info(f"⊙ {actual_date} 不是调仓日，下次调仓日：{self.strategy.get_next_rebalance_date(actual_date)}")

            # 5. 调用策略生成信号
            logger.info(f"\n[5/5] 生成交易信号...")

            # 创建一个简单的 portfolio 对象（仅用于接口兼容）
            class SimplePortfolio:
                def __init__(self):
                    self.positions = {}

                def get_position_count(self):
                    return len(self.positions)

            portfolio = SimplePortfolio()

            # 调用策略的 on_bar 方法
            signals = self.strategy.on_bar(
                data_handler=self.data_handler,
                current_date=actual_date,
                portfolio=portfolio
            )

            result['signals'] = signals
            result['signals_generated'] = len(signals)

            if signals:
                logger.info(f"✓ 生成 {len(signals)} 个交易信号")

                # 显示信号详情
                for signal in signals:
                    signal_type = "买入" if signal.signal_type == Signal.BUY else "卖出"
                    logger.info(f"  - {signal_type} {signal.code}: "
                          f"价格={signal.price:.2f}, "
                          f"权重={signal.weight:.2%}, "
                          f"原因={signal.reason}")

                # 导出信号到文件
                self._export_signals(signals, actual_date, export_format)

            else:
                logger.info(f"⊙ 没有生成交易信号")

            # 6. 更新状态
            self._update_state(actual_date, is_rebalance_day, signals)

            # 7. 返回结果
            if is_rebalance_day or signals:
                result['status'] = 'success'
                result['message'] = f'成功运行并生成 {len(signals)} 个信号'
            else:
                result['status'] = 'no_rebalance'
                result['message'] = f'运行完成，但不是调仓日'

            logger.info(f"\n{'='*70}")
            logger.info(f"实盘交易运行完成")
            logger.info(f"{'='*70}")

        except Exception as e:
            result['status'] = 'error'
            result['message'] = f'运行失败: {str(e)}'
            logger.error(f"\n✗ 错误: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return result

    def run_with_update(self,
                       target_date: date,
                       force_rebalance: bool = False,
                       export_format: str = "csv",
                       update_stocks: bool = True) -> Dict:
        """
        运行实盘交易流程并自动更新数据

        Args:
            target_date: 目标日期
            force_rebalance: 是否强制调仓
            export_format: 信号导出格式
            update_stocks: 是否更新股票数据

        Returns:
            执行结果字典（包含 update_result 和 run_result）
        """
        logger.info(f"\n{'#'*70}")
        logger.info(f"实盘交易完整流程（数据更新 + 策略运行）")
        logger.info(f"目标日期: {target_date}")
        logger.info(f"{'#'*70}")

        update_result = None

        # 1. 更新数据
        if update_stocks:
            logger.info(f"\n【第一步】更新数据...")
            logger.info(f"-" * 70)

            try:
                # 获取需要更新的股票列表
                stock_pool = self._get_stock_pool_for_update()

                if stock_pool:
                    # 批量更新股票数据
                    logger.info(f"准备更新 {len(stock_pool)} 只股票的数据...")
                    update_result = self.data_updater.update_batch_stock_data(
                        stock_codes=stock_pool,
                        end_date=target_date.strftime('%Y%m%d'),
                        delay=1.0
                    )
                else:
                    logger.warning(f"⚠ 没有找到需要更新的股票，跳过数据更新")
                    update_result = None

            except Exception as e:
                logger.error(f"✗ 数据更新失败: {e}")
                update_result = None

        # 2. 重新加载数据（清空旧的 data_handler）
        logger.info(f"\n【第二步】重新加载数据...")
        logger.info(f"-" * 70)
        self.data_handler = None

        # 3. 运行策略
        logger.info(f"\n【第三步】运行策略...")
        logger.info(f"-" * 70)
        run_result = self.run(
            target_date=target_date,
            force_rebalance=force_rebalance,
            export_format=export_format
        )

        logger.info(f"\n{'#'*70}")
        logger.info(f"完整流程执行完成")
        logger.info(f"{'#'*70}")

        return {
            'date': target_date,
            'update_result': update_result,
            'run_result': run_result
        }

    def _get_stock_pool_for_update(self) -> List[str]:
        """
        获取需要更新的股票池

        Returns:
            股票代码列表
        """
        # 如果策略指定了股票池，使用策略的股票池
        if 'stock_pool' in self.strategy.params and self.strategy.params['stock_pool'] is not None:
            return [str(code) for code in self.strategy.params['stock_pool']]

        # 否则使用 DataHandler 中的所有股票
        if self.data_handler is not None:
            return self.data_handler.get_all_codes()

        # 如果 DataHandler 还未加载，从数据目录扫描获取股票列表
        try:
            import glob
            import os

            # 扫描数据目录（优先使用 CSV）
            file_ext = "*.csv"  # 优先使用 CSV
            data_files = glob.glob(os.path.join(self.data_dir, file_ext))

            if not data_files:
                file_ext = "*.parquet"
                data_files = glob.glob(os.path.join(self.data_dir, file_ext))

            if data_files:
                # 从文件名提取股票代码
                stock_codes = []
                for file_path in data_files:
                    code = os.path.basename(file_path).replace('.parquet', '').replace('.csv', '')
                    stock_codes.append(code)

                logger.info(f"从数据目录扫描到 {len(stock_codes)} 只股票")
                return stock_codes
            else:
                logger.warning(f"⚠ 数据目录为空: {self.data_dir}")
                return []

        except Exception as e:
            logger.warning(f"⚠ 扫描数据目录失败: {e}")
            return []

    def _export_signals(self,
                       signals: List[Signal],
                       current_date: date,
                       export_format: str) -> None:
        """
        导出信号到文件

        Args:
            signals: 信号列表
            current_date: 当前日期
            export_format: 导出格式 ("csv", "json", 或 "both")
        """
        if not signals:
            return

        date_str = current_date.strftime('%Y%m%d')

        # 导出 CSV
        if export_format in ['csv', 'both']:
            csv_file = os.path.join(self.signal_output_dir, f"signals_{date_str}.csv")

            # 转换为 DataFrame
            data = []
            for signal in signals:
                data.append({
                    'date': signal.date.strftime('%Y-%m-%d') if signal.date else date_str,
                    'code': signal.code,
                    'signal_type': signal.signal_type,
                    'price': signal.price,
                    'weight': signal.weight,
                    'reason': signal.reason
                })

            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"  ✓ 信号已导出到: {csv_file}")

        # 导出 JSON
        if export_format in ['json', 'both']:
            json_file = os.path.join(self.signal_output_dir, f"signals_{date_str}.json")

            signal_data = {
                'date': date_str,
                'total_signals': len(signals),
                'buy_signals': len([s for s in signals if s.signal_type == Signal.BUY]),
                'sell_signals': len([s for s in signals if s.signal_type == Signal.SELL]),
                'signals': [signal.to_dict() for signal in signals]
            }

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(signal_data, f, ensure_ascii=False, indent=2)

            logger.info(f"  ✓ 信号已导出到: {json_file}")

    def _update_state(self,
                     current_date: date,
                     is_rebalance_day: bool,
                     signals: List[Signal]) -> None:
        """
        更新运行状态

        Args:
            current_date: 当前日期
            is_rebalance_day: 是否为调仓日
            signals: 生成的信号列表
        """
        # 更新交易日计数
        if self.state['last_trading_date'] != current_date.strftime('%Y-%m-%d'):
            self.state['trading_days_count'] += 1
            self.state['last_trading_date'] = current_date.strftime('%Y-%m-%d')

        # 更新调仓日期
        if is_rebalance_day:
            self.state['last_rebalance_date'] = current_date.strftime('%Y-%m-%d')

        # 记录执行历史
        execution_record = {
            'date': current_date.strftime('%Y-%m-%d'),
            'is_rebalance': is_rebalance_day,
            'signals': {
                'total': len(signals),
                'buy': len([s for s in signals if s.signal_type == Signal.BUY]),
                'sell': len([s for s in signals if s.signal_type == Signal.SELL]),
                'details': [s.to_dict() for s in signals]
            }
        }

        # 只保留最近 100 条记录
        self.state['execution_history'].append(execution_record)
        if len(self.state['execution_history']) > 100:
            self.state['execution_history'] = self.state['execution_history'][-100:]

        # 保存状态
        self._save_state()

    def _save_state(self) -> None:
        """保存状态到文件"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"⚠ 保存状态失败: {e}")

    def _load_state(self) -> None:
        """从文件加载状态"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self.state = json.load(f)
                logger.info(f"✓ 已加载历史状态文件")
        except Exception as e:
            logger.warning(f"⚠ 加载状态文件失败: {e}，将使用初始状态")

    def get_execution_summary(self) -> Dict:
        """
        获取执行摘要

        Returns:
            执行摘要字典
        """
        history = self.state['execution_history']

        if not history:
            return {
                'total_executions': 0,
                'rebalance_count': 0,
                'trading_days_count': self.state['trading_days_count'],
                'last_rebalance_date': self.state['last_rebalance_date'],
                'recent_signals': []
            }

        # 统计调仓次数
        rebalance_count = sum(1 for record in history if record['is_rebalance'])

        return {
            'total_executions': len(history),
            'rebalance_count': rebalance_count,
            'trading_days_count': self.state['trading_days_count'],
            'last_rebalance_date': self.state['last_rebalance_date'],
            'recent_signals': history[-10:]  # 最近10次执行
        }

    def print_execution_summary(self) -> None:
        """打印执行摘要"""
        summary = self.get_execution_summary()

        logger.info(f"\n{'='*70}")
        logger.info(f"执行摘要")
        logger.info(f"{'='*70}")
        logger.info(f"总执行次数: {summary['total_executions']}")
        logger.info(f"调仓次数: {summary['rebalance_count']}")
        logger.info(f"交易日计数: {summary['trading_days_count']}")
        logger.info(f"上次调仓日: {summary['last_rebalance_date'] or '未调仓'}")

        if summary['recent_signals']:
            logger.info(f"\n最近的执行记录:")
            for record in summary['recent_signals'][-5:]:
                logger.info(f"  {record['date']}: "
                      f"{'[调仓]' if record['is_rebalance'] else '[普通]'}, "
                      f"{record['signals']['total']} 个信号 "
                      f"(买入: {record['signals']['buy']}, 卖出: {record['signals']['sell']})")

        logger.info(f"{'='*70}")

    def reset_state(self) -> None:
        """重置状态"""
        self.state = {
            'strategy_start_date': None,
            'last_trading_date': None,
            'trading_days_count': 0,
            'last_rebalance_date': None,
            'execution_history': []
        }
        self._save_state()
        logger.info(f"✓ 状态已重置")

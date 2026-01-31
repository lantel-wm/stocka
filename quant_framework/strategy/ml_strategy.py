"""
机器学习策略
使用训练好的 LightGBM 模型进行预测和选股
"""

from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Callable
import pandas as pd
import numpy as np
from pathlib import Path

from ..data.data_handler import DataHandler
from .base_strategy import BaseStrategy, Signal
from ..model import LGBModel


class CSZScoreNorm:
    """
    Cross Sectional ZScore Normalization
    截面标准化：按日期分组进行 z-score 标准化
    """

    def __init__(self, factors: List[str], method: str = "zscore"):
        """
        初始化

        Args:
            method: 标准化方法，"zscore" 或 "robust"
        """
        if method == "zscore":
            self.zscore_func = self._zscore
        elif method == "robust":
            self.zscore_func = self._robust_zscore
        else:
            raise ValueError(f"不支持的标准化方法: {method}")

    def __call__(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        对指定列进行截面标准化

        Args:
            df: 输入数据（需要有 date 列或多级索引）
            columns: 需要标准化的列名列表

        Returns:
            标准化后的数据
        """
        # 如果 DataFrame 有 date 列（多级索引已重置）
        if 'date' in df.columns:
            df[columns] = df[columns].groupby("date", group_keys=False).apply(self.zscore_func)
        # 如果 DataFrame 是单日数据（索引是 code）
        else:
            # 直接对所有数据进行标准化
            df[columns] = df[columns].apply(self.zscore_func)

        return df

    def _zscore(self, x: pd.Series) -> pd.Series:
        """标准 z-score 标准化"""
        return (x - x.mean()) / x.std()

    def _robust_zscore(self, x: pd.Series) -> pd.Series:
        """鲁棒 z-score 标准化（使用中位数和 MAD）"""
        median = x.median()
        mad = np.median(np.abs(x - median))
        return (x - median) / (mad * 1.4826)  # 1.4826 是使得 MAD 与标准差可比的常数


class MLStrategy(BaseStrategy):
    """
    基于机器学习模型的选股策略

    策略逻辑：
    1. 每隔 N 个交易日使用模型对所有股票进行预测
    2. 选取预测分数最高的前 K 只股票
    3. 卖出不在新选中的持仓股票
    4. 买入新选中的股票（等权重或按预测分数加权）

    参数说明：
    - model_path: 模型文件路径（.pkl 或 .txt）
    - top_k: 选取预测分数最高的前 K 只股票（默认：30）
    - rebalance_days: 调仓周期（天数），每 N 个交易日调仓一次（默认：20）
    - weight_method: 仓位分配方式，'equal'等权重 或 'score'按分数加权（默认：'equal'）
    - min_score: 最低预测分数阈值，低于此值的股票不会被选中（默认：None）
    - hold_days: 持有天数，0 表示不限制（默认：0）
    - stock_pool: 股票池，None 表示使用全部股票（默认：None）
    - norm_method: 标准化方法，"zscore" 或 "robust"（默认："zscore"）
    - stop_loss: 止损阈值（百分比），如 0.05 表示亏损 5% 时止损（默认：None，不启用）
    - stop_loss_check_daily: 是否每日检查止损，False 表示只在调仓日检查（默认：True）
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        初始化机器学习策略

        Args:
            params: 策略参数字典
        """
        default_params = {
            'model_path': None,          # 模型路径（必需）
            'top_k': 10,                 # 选股数量
            'rebalance_days': 5,        # 调仓周期（交易日天数）
            'weight_method': 'equal',    # 仓位分配方式
            'min_score': None,           # 最低分数阈值
            'hold_days': 0,              # 最小持有天数
            'stock_pool': None,          # 股票池
            'norm_method': 'zscore',     # 标准化方法
            'stop_loss': 0.05,           # 止损阈值（百分比）
            'stop_loss_check_daily': True,  # 是否每日检查止损
        }

        if params:
            default_params.update(params)

        super().__init__(default_params)

        # 加载模型
        if self.params['model_path'] is None:
            raise ValueError("必须指定 model_path 参数")

        self.model = LGBModel.load_model(self.params['model_path'])
        self.factors = self.model.factors

        if not self.factors:
            raise ValueError("模型中没有 factors 信息，无法进行预测")

        # 初始化标准化器
        self.normalizer = CSZScoreNorm(factors=self.factors, method=self.params['norm_method'])

        print(f"✓ 模型加载成功")
        print(f"  - 因子数量: {len(self.factors)}")
        print(f"  - 选股数量: {self.params['top_k']}")
        print(f"  - 调仓周期: 每 {self.params['rebalance_days']} 个交易日")
        print(f"  - 标准化方法: {self.params['norm_method']}")
        if self.params['stop_loss'] is not None:
            print(f"  - 止损阈值: {self.params['stop_loss']*100:.2f}%")
            print(f"  - 止损检查: {'每日' if self.params['stop_loss_check_daily'] else '仅调仓日'}")

        # 跟踪状态
        self.last_rebalance_date = None  # 上次调仓日期
        self.trading_days_count = 0      # 交易日计数
        # position_entries: {code: {'entry_date': date, 'entry_price': float}}
        self.position_entries = {}       # 记录买入日期和买入价格

    def on_bar(self, data_handler, current_date: date, portfolio) -> List[Signal]:
        """
        每日运行策略逻辑

        Args:
            data_handler: 数据处理器对象
            current_date: 当前交易日期
            portfolio: 投资组合对象

        Returns:
            交易信号列表
        """
        signals = []

        # 增加交易日计数
        self.trading_days_count += 1

        # 检查止损（每日或调仓日）
        if self.params['stop_loss'] is not None:
            if self.params['stop_loss_check_daily'] or self._is_rebalance_day(current_date):
                stop_loss_signals = self._check_stop_loss(
                    data_handler, current_date, portfolio
                )
                signals.extend(stop_loss_signals)

        # 检查是否为调仓日
        if not self._is_rebalance_day(current_date):
            return signals

        # 获取待选股票池
        stock_pool = self._get_stock_pool(data_handler, current_date)

        # 准备预测数据
        pred_data = self._prepare_prediction_data(data_handler, current_date, stock_pool)
        
        if pred_data is None or len(pred_data) == 0:
            print(f"{current_date}: 没有可用的数据进行预测")
            return signals

        # 使用模型预测
        predictions = self.model.predict(pred_data)
        
        # 选股
        selected_stocks = self._select_stocks(predictions, pred_data)

        if not selected_stocks:
            print(f"{current_date}: 没有选中任何股票")
            return signals

        # 生成交易信号
        signals = self._generate_signals(
            current_date,
            selected_stocks,
            portfolio,
            data_handler
        )

        # 更新最后调仓日期
        self.last_rebalance_date = current_date

        print(f"{current_date}: 第 {self.trading_days_count} 个交易日，调仓！选中 {len(selected_stocks)} 只股票，生成 {len(signals)} 个信号")

        return signals

    def _is_rebalance_day(self, current_date: date) -> bool:
        """
        判断是否为调仓日

        Args:
            current_date: 当前日期

        Returns:
            是否为调仓日
        """
        rebalance_days = self.params['rebalance_days']

        # 第一次总是调仓
        if self.last_rebalance_date is None:
            return True

        # 计算距离上次调仓的交易日数
        if not hasattr(self, 'last_rebalance_day_count'):
            self.last_rebalance_day_count = 0

        days_since_last = self.trading_days_count - self.last_rebalance_day_count

        if days_since_last >= rebalance_days:
            self.last_rebalance_day_count = self.trading_days_count
            return True

        return False

    def _get_stock_pool(self, data_handler, current_date: date) -> List[str]:
        """
        获取股票池

        Args:
            data_handler: 数据处理器
            current_date: 当前日期

        Returns:
            股票代码列表
        """
        # 如果策略指定了股票池，使用策略的股票池
        if self.params['stock_pool'] is not None:
            return [str(code) for code in self.params['stock_pool']]

        # 否则使用全部股票
        return data_handler.get_all_codes()

    def _prepare_prediction_data(self, data_handler, current_date: date,
                                  stock_pool: List[str]) -> Optional[pd.DataFrame]:
        """
        准备预测所需的数据

        参考训练时的预处理：
        1. 截面标准化（Cross Sectional Z-Score Normalization）
        2. 填充缺失值为 0

        Args:
            data_handler: 数据处理器
            current_date: 当前日期
            stock_pool: 股票池

        Returns:
            准备好的预测数据
        """
        try:
            # 获取当前日期的数据
            daily_data = data_handler.get_daily_data(current_date)

            if daily_data is None or len(daily_data) == 0:
                return None

            # 过滤股票池
            daily_data = daily_data[daily_data.index.isin(stock_pool)]

            # 检查因子列是否存在
            missing_factors = set(self.factors) - set(daily_data.columns)
            if missing_factors:
                print(f"警告: 以下因子列不存在: {missing_factors}")
                return None

            # 复制数据，避免修改原始数据
            pred_data = daily_data.copy()

            # 过滤掉 close 为 NaN 的股票
            pred_data = pred_data[~pred_data['close'].isna()]

            if len(pred_data) == 0:
                print(f"{current_date}: 所有股票的 close 价格都为 NaN")
                return None

            # 截面标准化（Cross Sectional Z-Score Normalization）
            pred_data = self.normalizer(pred_data, self.factors)

            # 填充缺失值为 0
            pred_data[self.factors] = pred_data[self.factors].fillna(0)

            return pred_data

        except Exception as e:
            print(f"准备预测数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _select_stocks(self, predictions: pd.Series, pred_data: pd.DataFrame) -> List[Dict]:
        """
        根据预测分数选股

        Args:
            predictions: 预测分数 Series
            pred_data: 预测数据

        Returns:
            选中的股票列表，每个元素为 {code: str, score: float, weight: float}
        """
        top_k = self.params['top_k']
        min_score = self.params['min_score']
        weight_method = self.params['weight_method']

        # 创建结果 DataFrame
        result_df = pd.DataFrame({
            'code': predictions.index,
            'score': predictions.values
        })

        # 过滤低于最低分数的股票
        if min_score is not None:
            result_df = result_df[result_df['score'] >= min_score]

        if len(result_df) == 0:
            return []

        # 按分数降序排序
        result_df = result_df.sort_values('score', ascending=False)

        # 选取前 K 只
        result_df = result_df.head(top_k)

        # 计算权重
        if weight_method == 'equal':
            # 等权重
            result_df['weight'] = 1.0 / len(result_df)
        elif weight_method == 'score':
            # 按分数加权
            total_score = result_df['score'].sum()
            result_df['weight'] = result_df['score'] / total_score
        else:
            raise ValueError(f"不支持的权重分配方式: {weight_method}")

        # 转换为字典列表
        selected = result_df.to_dict('records')

        return selected

    def _generate_signals(self, current_date: date, selected_stocks: List[Dict],
                          portfolio, data_handler) -> List[Signal]:
        """
        生成交易信号

        Args:
            current_date: 当前日期
            selected_stocks: 选中的股票列表
            portfolio: 投资组合对象
            data_handler: 数据处理器

        Returns:
            交易信号列表
        """
        signals = []

        # 获取当前持仓
        current_positions = set(portfolio.positions.keys())

        # 选中新股票的代码集合
        selected_codes = set(stock['code'] for stock in selected_stocks)

        # 1. 卖出不在新选中的股票
        for code in current_positions:
            if code not in selected_codes:
                # 检查持有天数限制
                if self.params['hold_days'] > 0:
                    if code in self.position_entries:
                        entry_date = self.position_entries[code]['entry_date']
                        days_held = (current_date - entry_date).days
                        if days_held < self.params['hold_days']:
                            # 未达到最小持有天数，继续持有
                            continue

                # 获取当前价格
                price = self._get_current_price(data_handler, current_date, code)

                # 如果价格获取失败，跳过该股票
                if price is None:
                    print(f"{current_date}: 跳过卖出 {code}，无法获取价格")
                    continue

                # 生成卖出信号
                signal = Signal()
                signal.date = current_date
                signal.code = code
                signal.signal_type = Signal.SELL
                signal.price = price
                signal.weight = 1.0  # 全部卖出
                signal.reason = f"不在新选中的股票中"

                signals.append(signal)

                # 清除持仓记录
                if code in self.position_entries:
                    del self.position_entries[code]

        # 2. 买入新选中的股票
        for stock in selected_stocks:
            code = stock['code']
            target_weight = stock['weight']

            # 如果已经持有，不重复买入
            if code in current_positions:
                continue

            # 获取当前价格
            price = self._get_current_price(data_handler, current_date, code)

            # 如果价格获取失败，跳过该股票
            if price is None:
                print(f"{current_date}: 跳过买入 {code}，无法获取价格")
                continue

            # 生成买入信号
            signal = Signal()
            signal.date = current_date
            signal.code = code
            signal.signal_type = Signal.BUY
            signal.price = price
            signal.weight = target_weight
            signal.reason = f"模型预测分数: {stock['score']:.4f}"

            signals.append(signal)

            # 记录买入日期和买入价格
            self.position_entries[code] = {
                'entry_date': current_date,
                'entry_price': price
            }

        return signals

    def _get_current_price(self, data_handler: DataHandler, current_date: date, code: str) -> Optional[float]:
        """
        获取股票当前价格

        Args:
            data_handler: 数据处理器
            current_date: 当前日期
            code: 股票代码

        Returns:
            当前价格，如果获取失败返回 None
        """
        try:
            stock_data = data_handler.get_data_before(code, current_date)

            if stock_data is None or len(stock_data) == 0:
                return None

            # 获取最后一行的收盘价
            close_price = stock_data['close'].iloc[-1]

            # 检查价格是否为 NaN
            if pd.isna(close_price):
                return None

            return float(close_price)

        except Exception:
            return None

    def _check_stop_loss(self, data_handler: DataHandler, current_date: date, portfolio) -> List[Signal]:
        """
        检查并执行止损

        Args:
            data_handler: 数据处理器
            current_date: 当前日期
            portfolio: 投资组合对象

        Returns:
            止损信号列表
        """
        signals = []
        stop_loss_threshold = self.params['stop_loss']

        # 获取当前持仓
        current_positions = portfolio.positions

        # 遍历所有持仓，检查是否需要止损
        for code, position in current_positions.items():
            # 检查是否有买入记录
            if code not in self.position_entries:
                continue

            entry_info = self.position_entries[code]
            entry_price = entry_info['entry_price']

            # 获取当前价格
            current_price = self._get_current_price(data_handler, current_date, code)

            if current_price is None:
                continue

            # 计算收益率
            return_rate = (current_price - entry_price) / entry_price

            # 检查是否触发止损
            if return_rate <= -stop_loss_threshold:
                # 生成卖出信号
                signal = Signal()
                signal.date = current_date
                signal.code = code
                signal.signal_type = Signal.SELL
                signal.price = current_price
                signal.weight = 1.0  # 全部卖出
                signal.reason = f"止损触发: 买入价={entry_price:.2f}, 当前价={current_price:.2f}, 收益率={return_rate*100:.2f}%"

                signals.append(signal)

                # 清除持仓记录
                del self.position_entries[code]

                print(f"{current_date}: 止损卖出 {code}，买入价={entry_price:.2f}, "
                      f"当前价={current_price:.2f}, 亏损={return_rate*100:.2f}%")

        return signals

    def reset(self) -> None:
        """重置策略状态"""
        super().reset()
        self.last_rebalance_date = None
        self.trading_days_count = 0
        self.position_entries = {}
        if hasattr(self, 'last_rebalance_day_count'):
            delattr(self, 'last_rebalance_day_count')

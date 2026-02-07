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
from ..factor.alpha158 import Alpha158
from ..utils.logger import get_logger

logger = get_logger(__name__)


def _calculate_single_stock_factors(stock_data, factor_calculator, prediction_ts, code, min_history_length):
    """
    计算单只股票的因子（独立函数，可在多进程中调用）

    Args:
        stock_data: 股票历史数据 (DataFrame with date index)
        factor_calculator: Alpha158 因子计算器实例
        prediction_ts: 预测日期（pd.Timestamp）
        code: 股票代码
        min_history_length: 最小历史数据长度要求

    Returns:
        tuple: (code, factor_data) 或 (code, None) 如果失败
            - code: 股票代码
            - factor_data: 包含因子的 DataFrame (如果成功) 或 None (如果失败)
    """
    try:
        # 检查历史数据是否足够
        if len(stock_data) < min_history_length:
            return (code, None)

        # 检查是否有 prediction_date 的数据
        if prediction_ts not in stock_data.index:
            return (code, None)

        # 重置索引，使 date 成为列
        stock_data_reset = stock_data.reset_index()

        # 计算因子
        factor_data = factor_calculator.calculate(stock_data_reset)

        # 提取 prediction_ts 的数据
        daily_factor = factor_data[factor_data['date'] == prediction_ts]

        if len(daily_factor) == 0:
            return (code, None)

        # 设置股票代码
        daily_factor = daily_factor.copy()
        daily_factor['stock_code'] = code

        return (code, daily_factor)

    except Exception:
        return (code, None)


class CSZScoreNorm:
    """
    Cross Sectional ZScore Normalization
    截面标准化: 按日期分组进行 z-score 标准化
    """

    def __init__(self, factors: List[str], method: str = "zscore"):
        """
        初始化

        Args:
            method: 标准化方法, "zscore" 或 "robust"
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
            df: 输入数据 (需要有 date 列或多级索引)
            columns: 需要标准化的列名列表

        Returns:
            标准化后的数据
        """
        # 如果 DataFrame 有 date 列 (多级索引已重置)
        if 'date' in df.columns:
            df[columns] = df[columns].groupby("date", group_keys=False).apply(self.zscore_func)
        # 如果 DataFrame 是单日数据 (索引是 code)
        else:
            # 直接对所有数据进行标准化
            df[columns] = df[columns].apply(self.zscore_func)

        return df

    def _zscore(self, x: pd.Series) -> pd.Series:
        """标准 z-score 标准化"""
        return (x - x.mean()) / x.std()

    def _robust_zscore(self, x: pd.Series) -> pd.Series:
        """鲁棒 z-score 标准化 (使用中位数和 MAD)"""
        median = x.median()
        mad = np.median(np.abs(x - median))
        return (x - median) / (mad * 1.4826)  # 1.4826 是使得 MAD 与标准差可比的常数


class MLStrategy(BaseStrategy):
    """
    基于机器学习模型的选股策略

    策略逻辑:
    1. 每隔 N 个交易日使用模型对所有股票进行预测
    2. 选取预测分数最高的前 K 只股票
    3. 从 top K 中过滤掉预测分数低于阈值的股票（如果设置了 min_score）
    4. 卖出不在新选中的持仓股票
    5. 买入新选中的股票 (等权重或按预测分数加权)

    参数说明:
    - model_path: 模型文件路径 (.pkl 或 .txt)
    - top_k: 选取预测分数最高的前 K 只股票 (默认: 10)
    - rebalance_days: 调仓周期 (天数), 每 N 个交易日调仓一次 (默认: 5)
    - weight_method: 仓位分配方式, 'equal'等权重 或 'score'按分数加权 (默认: 'equal')
    - min_score: 最低预测分数阈值, 从 top_k 中过滤掉低于此值的股票 (默认: None, 不启用)
                  例如设置为 0.01，则只持有 top_k 中预测分数 >= 0.01 的股票
    - hold_days: 持有天数, 0 表示不限制 (默认: 0)
    - stock_pool: 股票池, None 表示使用全部股票 (默认: None)
    - norm_method: 标准化方法, "zscore" 或 "robust" (默认: "zscore")
    - stop_loss: 止损阈值 (百分比), 如 0.05 表示亏损 5% 时止损 (默认: 0.05)
    - stop_loss_check_daily: 是否每日检查止损, False 表示只在调仓日检查 (默认: True)
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        初始化机器学习策略

        Args:
            params: 策略参数字典
        """
        default_params = {
            'model_path': None,          # 模型路径 (必需)
            'top_k': 10,                 # 选股数量
            'rebalance_days': 5,        # 调仓周期 (交易日天数)
            'weight_method': 'equal',    # 仓位分配方式
            'min_score': None,           # 最低分数阈值
            'hold_days': 0,              # 最小持有天数
            'stock_pool': None,          # 股票池
            'norm_method': 'zscore',     # 标准化方法
            'stop_loss': 0.05,           # 止损阈值 (百分比)
            'stop_loss_check_daily': True,  # 是否每日检查止损
            'trailing_stop_percent': None,   # 移动止损百分比 (如 0.05 = 5%)
            'trailing_stop_activation': 0.0, # 移动止损激活阈值 (如 0.03 = 盈利3%后启用)
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
            raise ValueError("模型中没有 factors 信息, 无法进行预测")

        # 初始化标准化器
        self.normalizer = CSZScoreNorm(factors=self.factors, method=self.params['norm_method'])

        # 初始化因子计算器（用于实盘时实时计算因子）
        self.factor_calculator = Alpha158()
        # Alpha158 最长窗口为 60 天，需要至少 60 天的历史数据
        self.min_history_length = 60

        logger.info(f"模型加载成功")
        logger.info(f"  - 因子数量: {len(self.factors)}")
        logger.info(f"  - 选股数量 (top_k): {self.params['top_k']}")
        logger.info(f"  - 调仓周期: 每 {self.params['rebalance_days']} 个交易日")
        logger.info(f"  - 仓位分配: {self.params['weight_method']}")
        logger.info(f"  - 标准化方法: {self.params['norm_method']}")
        if self.params['min_score'] is not None:
            logger.info(f"  - 预测分数阈值: >= {self.params['min_score']:.4f}")
        else:
            logger.info(f"  - 预测分数阈值: 未启用")
        if self.params['stop_loss'] is not None:
            logger.info(f"  - 固定止损阈值: {self.params['stop_loss']*100:.2f}%")
            logger.info(f"  - 止损检查: {'每日' if self.params['stop_loss_check_daily'] else '仅调仓日'}")
        if self.params['trailing_stop_percent'] is not None:
            logger.info(f"  - 移动止损: {self.params['trailing_stop_percent']*100:.2f}%")
            logger.info(f"  - 移动止损激活阈值: {self.params['trailing_stop_activation']*100:.2f}%")
        else:
            logger.info(f"  - 移动止损: 未启用")

        # 跟踪状态
        self.last_rebalance_date = None  # 上次调仓日期
        self.trading_days_count = 0      # 交易日计数
        # position_entries: {code: {'entry_date': date, 'entry_price': float}}
        self.position_entries = {}       # 记录买入日期和买入价格

        # 实盘模式支持
        self.trading_dates = []          # 交易日历列表
        self.strategy_start_date = None  # 策略开始运行日期

        # 预测记录（用于回测分析）
        # 格式: [{'date': date, 'predictions': [(code, score), ...]}, ...]
        self.prediction_history = []    # 每次预测的前20名股票记录

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

        # 检查止损 (每日或调仓日)
        if self.params['stop_loss'] is not None or self.params['trailing_stop_percent'] is not None:
            if self.params['stop_loss_check_daily'] or self._is_rebalance_day(current_date):
                # 先更新持仓的最高价格（用于移动止损）
                self._update_highest_price(data_handler, current_date)
                # 再检查止损
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
            logger.warning(f"{current_date}: 没有可用的数据进行预测")
            return signals

        # 使用模型预测
        predictions = self.model.predict(pred_data)

        # 保存前20名的预测结果
        self._save_top_predictions(current_date, predictions)

        # 选股
        selected_stocks = self._select_stocks(predictions, pred_data)

        if not selected_stocks:
            logger.warning(f"{current_date}: 没有选中任何股票，将清仓所有持仓")

        # 生成交易信号（即使没有选中股票，也会卖出所有当前持仓）
        signals = self._generate_signals(
            current_date,
            selected_stocks,
            portfolio,
            data_handler
        )

        # 更新最后调仓日期
        self.last_rebalance_date = current_date

        logger.info(f"{current_date}: 第 {self.trading_days_count} 个交易日, 调仓！选中 {len(selected_stocks)} 只股票, 生成 {len(signals)} 个信号")

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
        # 如果策略指定了股票池, 使用策略的股票池
        if self.params['stock_pool'] is not None:
            return [str(code) for code in self.params['stock_pool']]

        # 否则使用全部股票
        return data_handler.get_all_codes()

    def _prepare_prediction_data(self, data_handler: DataHandler, current_date: date,
                                  stock_pool: List[str]) -> Optional[pd.DataFrame]:
        """
        准备预测所需的数据

        重要：使用 current_date 前一个交易日的数据进行预测，避免未来函数（look-ahead bias）

        参考训练时的预处理:
        1. 截面标准化 (Cross Sectional Z-Score Normalization)
        2. 填充缺失值为 0

        Args:
            data_handler: 数据处理器
            current_date: 当前日期（将使用前一个交易日的数据）
            stock_pool: 股票池

        Returns:
            准备好的预测数据
        """
        try:
            # 获取前一个交易日的日期（用于预测当前日期，避免未来函数）
            prediction_date = data_handler.get_previous_trading_date(current_date, n=1)

            if prediction_date is None:
                logger.warning(f"{current_date}: 没有前一个交易日的数据，无法进行预测")
                return None

            # 获取前一个交易日的数据
            daily_data = data_handler.get_daily_data(prediction_date)

            if daily_data is None or len(daily_data) == 0:
                return None

            # 过滤股票池
            daily_data = daily_data[daily_data.index.isin(stock_pool)]

            # 检查因子列是否存在
            missing_factors = set(self.factors) - set(daily_data.columns)

            if missing_factors:
                # 因子不存在，需要实时计算
                logger.info(f"{current_date} (使用{prediction_date}的数据): 检测到 {len(missing_factors)} 个因子列不存在，开始实时计算...")
                pred_data = self._calculate_factors_realtime(
                    data_handler, prediction_date, stock_pool
                )
            else:
                # 因子已存在，直接使用
                pred_data = daily_data.copy()

            if pred_data is None or len(pred_data) == 0:
                return None

            # 过滤掉 close 为 NaN 的股票
            pred_data = pred_data[~pred_data['close'].isna()]

            if len(pred_data) == 0:
                logger.warning(f"{current_date} (使用{prediction_date}的数据): 所有股票的 close 价格都为 NaN")
                return None

            # 截面标准化 (Cross Sectional Z-Score Normalization)
            pred_data = self.normalizer(pred_data, self.factors)

            # 填充缺失值为 0
            pred_data[self.factors] = pred_data[self.factors].fillna(0)

            return pred_data

        except Exception as e:
            logger.error(f"准备预测数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_factors_realtime(self, data_handler: DataHandler, prediction_date: date,
                                    stock_pool: List[str]) -> Optional[pd.DataFrame]:
        """
        实时计算因子值（用于实盘交易）- 多进程并行版本

        使用多进程并行计算所有股票的因子，显著提升性能

        Args:
            data_handler: 数据处理器
            prediction_date: 预测日期（使用该日期的数据进行预测）
            stock_pool: 股票池

        Returns:
            包含因子值的 DataFrame，索引为股票代码
        """
        import time
        import multiprocessing as mp
        from multiprocessing import Pool
        from tqdm import tqdm

        start_time = time.time()
        prediction_ts = pd.Timestamp(prediction_date)

        # 确定进程数
        num_cpus = mp.cpu_count()
        num_processes = max(1, num_cpus - 1)  # 保留一个核心给系统

        logger.info(f"开始为 {len(stock_pool)} 只股票并行计算因子（多进程模式）...")
        logger.info(f"  - 预测日期: {prediction_date}")
        logger.info(f"  - CPU核心数: {num_cpus}")
        logger.info(f"  - 使用进程数: {num_processes}")

        # 第一步：批量加载所有股票的历史数据
        logger.info(f"[1/3] 批量加载股票历史数据...")
        load_start = time.time()

        tasks = []  # 存储 (stock_data, factor_calculator, prediction_ts, code, min_history_length)
        skipped_count = 0
        loaded_count = 0

        for code in stock_pool:
            try:
                stock_data = data_handler.get_data_before(code, prediction_date)

                if stock_data is None or len(stock_data) == 0:
                    skipped_count += 1
                    continue

                # 检查历史数据是否足够
                if len(stock_data) < self.min_history_length:
                    skipped_count += 1
                    continue

                # 检查是否有 prediction_date 的数据
                if prediction_ts not in stock_data.index:
                    skipped_count += 1
                    continue

                # 准备任务参数（使用 starmap 时会自动解包）
                tasks.append((stock_data, self.factor_calculator, prediction_ts, code, self.min_history_length))
                loaded_count += 1

            except Exception:
                skipped_count += 1
                continue

        load_time = time.time() - load_start
        logger.info(f"  ✓ 加载完成: {loaded_count} 只股票成功, {skipped_count} 只跳过 (耗时: {load_time:.2f}秒)")

        if not tasks:
            logger.error(f"没有成功加载任何股票的数据")
            return None

        # 第二步：多进程并行计算因子
        logger.info(f"[2/3] 并行计算 Alpha158 因子（{num_processes} 进程）...")
        calc_start = time.time()

        all_factor_data = []

        try:
            with Pool(processes=num_processes) as pool:
                # 使用 starmap 自动解包参数
                # imap 可以显示进度，但不会按顺序返回
                results = list(tqdm(
                    pool.starmap(_calculate_single_stock_factors, tasks),
                    total=len(tasks),
                    desc="计算因子",
                    unit="股"
                ))

            # 分离成功和失败的结果
            for code, factor_data in results:
                if factor_data is not None:
                    all_factor_data.append(factor_data)

        except Exception as e:
            logger.error(f"多进程计算出错: {e}")
            import traceback
            traceback.print_exc()
            return None

        calc_time = time.time() - calc_start
        success_count = len(all_factor_data)
        logger.info(f"  ✓ 因子计算完成: {success_count} 只股票成功 (耗时: {calc_time:.2f}秒)")

        if not all_factor_data:
            logger.error(f"没有成功计算任何股票的因子")
            return None

        # 第三步：合并结果
        logger.info(f"[3/3] 合并结果...")
        merge_start = time.time()

        # 合并所有因子数据
        all_factors = pd.concat(all_factor_data, ignore_index=True)

        # 设置索引为股票代码
        pred_data = all_factors.set_index('stock_code')
        pred_data.index.name = 'code'

        # 删除 date 列（已经不需要了）
        if 'date' in pred_data.columns:
            pred_data = pred_data.drop(columns=['date'])

        merge_time = time.time() - merge_start
        logger.info(f"  ✓ 合并完成 (耗时: {merge_time:.2f}秒)")

        # 总耗时统计
        total_time = time.time() - start_time
        logger.info(f"✓ 多进程因子计算完成！")
        logger.info(f"  - 总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
        logger.info(f"  - 成功: {len(pred_data)} 只")
        logger.info(f"  - 跳过: {skipped_count} 只")
        logger.info(f"  - 平均速度: {total_time/len(pred_data):.3f}秒/股")
        logger.info(f"  - 理论加速比: {num_processes}x")
        logger.info(f"  - 实际加速比: {9.64 / (total_time/len(pred_data)):.1f}x (相比原版9.64秒/股)")

        return pred_data

    def _save_top_predictions(self, current_date: date, predictions: pd.Series, top_n: int = 20) -> None:
        """
        保存前N名的预测结果

        Args:
            current_date: 当前日期
            predictions: 预测分数 Series (index为股票代码)
            top_n: 保存前N名 (默认: 20)
        """
        try:
            # 创建 DataFrame 并按分数降序排序
            pred_df = pd.DataFrame({
                'code': predictions.index,
                'score': predictions.values
            }).sort_values('score', ascending=False)

            # 取前 top_n 名
            top_predictions = pred_df.head(top_n)

            # 转换为列表格式
            predictions_list = [(row['code'], row['score']) for _, row in top_predictions.iterrows()]

            # 保存到历史记录
            self.prediction_history.append({
                'date': current_date,
                'predictions': predictions_list
            })

            logger.debug(f"{current_date}: 已保存前 {len(predictions_list)} 名预测结果")

        except Exception as e:
            logger.warning(f"保存预测结果时出错: {e}")

    def _select_stocks(self, predictions: pd.Series, pred_data: pd.DataFrame) -> List[Dict]:
        """
        根据预测分数选股

        选股逻辑：
        1. 先选出预测分数最高的 top_k 只股票
        2. 从 top_k 中过滤掉预测分数低于 min_score 的股票
        3. 对过滤后的股票分配权重

        Args:
            predictions: 预测分数 Series
            pred_data: 预测数据

        Returns:
            选中的股票列表, 每个元素为 {code: str, score: float, weight: float}
        """
        top_k = self.params['top_k']
        min_score = self.params['min_score']
        weight_method = self.params['weight_method']

        # 创建结果 DataFrame
        result_df = pd.DataFrame({
            'code': predictions.index,
            'score': predictions.values
        })

        # 按分数降序排序
        result_df = result_df.sort_values('score', ascending=False)

        # 先选取前 K 只
        top_k_df = result_df.head(top_k)

        # 从 top_k 中过滤低于最低分数的股票
        if min_score is not None:
            filtered_df = top_k_df[top_k_df['score'] >= min_score].copy()
        else:
            filtered_df = top_k_df.copy()

        if len(filtered_df) == 0:
            logger.info(f"  - Top {top_k} 只股票中，没有预测分数 >= {min_score} 的股票，本次不持仓")
            return []

        filtered_count = len(top_k_df) - len(filtered_df)
        if filtered_count > 0:
            logger.info(f"  - Top {top_k} 中有 {filtered_count} 只股票因预测分数低于阈值 {min_score} 被过滤")

        # 计算权重
        if weight_method == 'equal':
            # 等权重
            filtered_df = filtered_df.copy()
            # filtered_df['weight'] = 1.0 / len(filtered_df)
            # 过滤后不要全仓买入
            filtered_df['weight'] = 1.0 / top_k
        elif weight_method == 'score':
            # 按分数加权
            total_score = filtered_df['score'].sum()
            filtered_df = filtered_df.copy()
            filtered_df['weight'] = filtered_df['score'] / total_score
        else:
            raise ValueError(f"不支持的权重分配方式: {weight_method}")

        # 转换为字典列表
        selected = filtered_df.to_dict('records')

        logger.info(f"  - 最终选中 {len(selected)} 只股票 (从 top {top_k} 中筛选，阈值: {min_score})")

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
                            # 未达到最小持有天数, 继续持有
                            continue

                # 获取当前价格
                price = self._get_current_price(data_handler, current_date, code)

                # 如果价格获取失败, 跳过该股票
                if price is None:
                    logger.warning(f"{current_date}: 跳过卖出 {code}, 无法获取价格")
                    continue

                # 计算盈亏率（如果有买入记录）
                if code in self.position_entries:
                    entry_price = self.position_entries[code]['entry_price']
                    return_rate = (price - entry_price) / entry_price
                    reason = f"不在新选中的股票中: 买入价={entry_price:.2f}, 当前价={price:.2f}, 收益率={return_rate*100:.2f}%"
                else:
                    reason = "不在新选中的股票中"

                # 生成卖出信号
                signal = Signal()
                signal.date = current_date
                signal.code = code
                signal.signal_type = Signal.SELL
                signal.price = price
                signal.weight = 1.0  # 全部卖出
                signal.reason = reason

                signals.append(signal)

                # 清除持仓记录
                if code in self.position_entries:
                    del self.position_entries[code]

        # 2. 买入新选中的股票
        for stock in selected_stocks:
            code = stock['code']
            target_weight = stock['weight']

            # 如果已经持有, 不重复买入
            if code in current_positions:
                continue

            # 获取当前价格
            price = self._get_current_price(data_handler, current_date, code)

            # 如果价格获取失败, 跳过该股票
            if price is None:
                logger.warning(f"{current_date}: 跳过买入 {code}, 无法获取价格")
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

            # 记录买入日期、买入价格和最高价格（用于移动止损）
            self.position_entries[code] = {
                'entry_date': current_date,
                'entry_price': price,
                'highest_price': price  # 初始化最高价格为买入价
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
            当前价格, 如果获取失败返回 None
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

    def _update_highest_price(self, data_handler: DataHandler, current_date: date) -> None:
        """
        更新持仓的最高价格（用于移动止损）

        Args:
            data_handler: 数据处理器
            current_date: 当前日期
        """
        for code in list(self.position_entries.keys()):
            try:
                current_price = self._get_current_price(data_handler, current_date, code)

                if current_price is not None:
                    entry = self.position_entries[code]

                    # 更新最高价格（只能上移，不能下移）
                    if current_price > entry.get('highest_price', entry['entry_price']):
                        entry['highest_price'] = current_price

            except Exception:
                # 获取价格失败，跳过
                continue

    def _check_stop_loss(self, data_handler: DataHandler, current_date: date, portfolio) -> List[Signal]:
        """
        检查并执行止损（支持固定止损和移动止损）

        止损逻辑：
        1. 如果启用移动止损且盈利超过激活阈值：
           使用移动止损：stop_price = highest_price * (1 - trailing_stop_percent)
        2. 否则如果启用固定止损：
           使用固定止损：stop_price = entry_price * (1 - stop_loss)
        3. 如果 current_price < stop_price，触发止损

        Args:
            data_handler: 数据处理器
            current_date: 当前日期
            portfolio: 投资组合对象

        Returns:
            止损信号列表
        """
        signals = []
        stop_loss_threshold = self.params['stop_loss']
        trailing_stop_percent = self.params['trailing_stop_percent']
        trailing_stop_activation = self.params['trailing_stop_activation']

        # 如果没有启用任何止损，直接返回
        if stop_loss_threshold is None and trailing_stop_percent is None:
            return signals

        # 获取当前持仓
        current_positions = portfolio.positions

        # 遍历所有持仓, 检查是否需要止损
        for code, position in current_positions.items():
            # 检查是否有买入记录
            if code not in self.position_entries:
                continue

            entry_info = self.position_entries[code]
            entry_price = entry_info['entry_price']
            highest_price = entry_info.get('highest_price', entry_price)

            # 获取当前价格
            current_price = self._get_current_price(data_handler, current_date, code)

            if current_price is None:
                continue

            # 计算收益率
            return_rate = (current_price - entry_price) / entry_price

            # 确定止损价格
            stop_price = None
            stop_type = None

            # 1. 检查是否启用移动止损且盈利超过激活阈值
            if trailing_stop_percent is not None and return_rate >= trailing_stop_activation:
                # 使用移动止损
                stop_price = highest_price * (1 - trailing_stop_percent)
                stop_type = "移动止损"
            # 2. 否则使用固定止损（如果启用）
            elif stop_loss_threshold is not None:
                stop_price = entry_price * (1 - stop_loss_threshold)
                stop_type = "固定止损"

            # 检查是否触发止损
            if stop_price is not None and current_price < stop_price:
                # 生成卖出信号
                signal = Signal()
                signal.date = current_date
                signal.code = code
                signal.signal_type = Signal.SELL
                signal.price = current_price
                signal.weight = 1.0  # 全部卖出

                # 构建止损原因
                if trailing_stop_percent is not None and return_rate >= trailing_stop_activation:
                    signal.reason = (f"{stop_type}触发: 买入价={entry_price:.2f}, 最高价={highest_price:.2f}, "
                                   f"当前价={current_price:.2f}, 收益率={return_rate*100:.2f}%")
                else:
                    signal.reason = (f"{stop_type}触发: 买入价={entry_price:.2f}, "
                                   f"当前价={current_price:.2f}, 收益率={return_rate*100:.2f}%")

                signals.append(signal)

                # 清除持仓记录
                del self.position_entries[code]

                logger.info(f"{current_date}: {stop_type}卖出 {code}, 买入价={entry_price:.2f}, "
                          f"最高价={highest_price:.2f}, 当前价={current_price:.2f}, "
                          f"收益率={return_rate*100:.2f}%")

        return signals

    def reset(self) -> None:
        """重置策略状态"""
        super().reset()
        self.last_rebalance_date = None
        self.trading_days_count = 0
        self.position_entries = {}
        self.prediction_history = []  # 清空预测历史

    def initialize_for_live_trading(self,
                                   data_handler,
                                   strategy_start_date: date) -> None:
        """
        初始化实盘交易模式

        Args:
            data_handler: 数据处理器对象
            strategy_start_date: 策略开始运行的日期
        """
        self.strategy_start_date = strategy_start_date

        # 获取交易日历
        # 从策略开始日期到当前可用的所有交易日
        all_dates = data_handler.get_available_dates(
            start_date=strategy_start_date,
            end_date=date.today()
        )

        if not all_dates:
            raise ValueError(f"无法获取从 {strategy_start_date} 开始的交易日历")

        # 过滤并排序交易日历
        self.trading_dates = sorted([d for d in all_dates if d >= strategy_start_date])

        logger.info(f"实盘模式初始化完成:")
        logger.info(f"  - 策略开始日期: {strategy_start_date}")
        logger.info(f"  - 交易日历数量: {len(self.trading_dates)} 个交易日")
        logger.info(f"  - 交易日期范围: {self.trading_dates[0]} 至 {self.trading_dates[-1]}")

    def is_rebalance_day(self, current_date: date) -> bool:
        """
        判断给定日期是否为调仓日（实盘模式）

        Args:
            current_date: 当前日期

        Returns:
            是否为调仓日
        """
        if not self.trading_dates:
            # 如果没有初始化交易日历，使用回测模式的逻辑
            return self._is_rebalance_day(current_date)

        # 检查当前日期是否在交易日历中
        if current_date not in self.trading_dates:
            return False

        # 找到当前日期在交易日历中的索引
        try:
            current_index = self.trading_dates.index(current_date)
        except ValueError:
            return False

        # 计算从策略开始到当前日期的交易日数量
        trading_days_since_start = current_index + 1  # 索引从0开始，所以+1

        rebalance_days = self.params['rebalance_days']

        # 判断是否是调仓日
        # 例如：rebalance_days=5，则第1, 6, 11, 16...个交易日是调仓日
        if trading_days_since_start % rebalance_days == 1:
            return True

        return False

    def get_next_rebalance_date(self, current_date: date) -> Optional[date]:
        """
        获取下一个调仓日（实盘模式）

        Args:
            current_date: 当前日期

        Returns:
            下一个调仓日，如果不存在返回None
        """
        if not self.trading_dates:
            return None

        # 找到当前日期之后的交易日
        future_dates = [d for d in self.trading_dates if d > current_date]

        if not future_dates:
            return None

        rebalance_days = self.params['rebalance_days']

        # 获取当前日期的索引（如果不在交易日历中，找到最近的）
        try:
            current_index = self.trading_dates.index(current_date)
        except ValueError:
            # 找到第一个大于当前日期的索引
            for i, d in enumerate(self.trading_dates):
                if d > current_date:
                    current_index = i - 1
                    break
            else:
                current_index = len(self.trading_dates) - 1

        # 找下一个调仓日
        # 当前索引之后的每个 (rebalance_days - 1) 的倍数位置
        offset = rebalance_days - (current_index + 1) % rebalance_days
        if offset == rebalance_days:
            offset = 0

        next_index = current_index + offset
        if offset == 0:
            next_index += rebalance_days

        if next_index < len(self.trading_dates):
            return self.trading_dates[next_index]

        return None

    def get_rebalance_info(self, current_date: date) -> Dict:
        """
        获取当前日期的调仓信息

        Args:
            current_date: 当前日期

        Returns:
            调仓信息字典:
            {
                'is_trading_day': bool,           # 是否为交易日
                'is_rebalance_day': bool,         # 是否为调仓日
                'days_since_start': int,          # 距离策略开始的交易日数
                'days_until_next_rebalance': int, # 距离下次调仓的交易日数
                'next_rebalance_date': date,      # 下次调仓日期
                'last_rebalance_date': date,      # 上次调仓日期
            }
        """
        info = {
            'is_trading_day': False,
            'is_rebalance_day': False,
            'days_since_start': 0,
            'days_until_next_rebalance': 0,
            'next_rebalance_date': None,
            'last_rebalance_date': None,
        }

        if not self.trading_dates:
            return info

        # 检查是否为交易日
        is_trading_day = current_date in self.trading_dates
        info['is_trading_day'] = is_trading_day

        if not is_trading_day:
            return info

        # 获取索引
        current_index = self.trading_dates.index(current_date)
        info['days_since_start'] = current_index + 1

        # 检查是否为调仓日
        info['is_rebalance_day'] = self.is_rebalance_day(current_date)

        # 计算上次调仓日
        rebalance_days = self.params['rebalance_days']
        if current_index >= rebalance_days - 1:
            last_index = current_index - (current_index % rebalance_days)
            if last_index >= 0:
                info['last_rebalance_date'] = self.trading_dates[last_index]

        # 计算下次调仓日
        next_rebalance = self.get_next_rebalance_date(current_date)
        info['next_rebalance_date'] = next_rebalance

        if next_rebalance:
            try:
                next_index = self.trading_dates.index(next_rebalance)
                info['days_until_next_rebalance'] = next_index - current_index
            except ValueError:
                pass

        return info

    def get_prediction_history(self) -> List[Dict]:
        """
        获取预测历史记录

        Returns:
            预测历史记录列表，每个元素格式:
            {
                'date': date,
                'predictions': [(code, score), ...]  # 前20名
            }
        """
        return self.prediction_history

    def get_prediction_history_df(self) -> pd.DataFrame:
        """
        获取预测历史记录的DataFrame格式

        Returns:
            DataFrame，包含列: date, rank, code, score
        """
        if not self.prediction_history:
            return pd.DataFrame(columns=['date', 'rank', 'code', 'score'])

        # 展开预测历史
        records = []
        for record in self.prediction_history:
            pred_date = record['date']
            for rank, (code, score) in enumerate(record['predictions'], start=1):
                records.append({
                    'date': pred_date,
                    'rank': rank,
                    'code': code,
                    'score': score
                })

        return pd.DataFrame(records)

    def save_prediction_history(self, file_path: str) -> None:
        """
        保存预测历史到CSV文件

        Args:
            file_path: 保存路径 (例如: 'predictions.csv')
        """
        df = self.get_prediction_history_df()

        if not df.empty:
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"预测历史已保存到: {file_path}")
            logger.info(f"  - 记录数: {len(df)}")
            logger.info(f"  - 日期范围: {df['date'].min()} 至 {df['date'].max()}")
        else:
            logger.warning("没有预测历史记录可保存")

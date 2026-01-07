"""
投资组合管理模块
管理仓位、资金和风险控制
"""

from datetime import date, datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from .position import Position
from ..strategy.base_strategy import Signal


@dataclass
class Trade:
    """交易记录"""
    date: date
    code: str
    action: str  # 'buy' or 'sell'
    shares: int
    price: float
    amount: float
    commission: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            'date': self.date.strftime('%Y-%m-%d'),
            'code': self.code,
            'action': self.action,
            'shares': self.shares,
            'price': self.price,
            'amount': self.amount,
            'commission': self.commission,
            'reason': self.reason
        }


class Portfolio:
    """
    投资组合类
    管理资金、持仓和交易记录
    """

    def __init__(self,
                 initial_capital: float = 1000000.0,
                 max_single_position_ratio: float = 0.3):
        """
        初始化投资组合

        Args:
            initial_capital: 初始资金
            max_single_position_ratio: 单只股票最大仓位比例
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_single_position_ratio = max_single_position_ratio

        # 持仓管理 {code: Position}
        self.positions: Dict[str, Position] = {}

        # 交易历史
        self.trades: List[Trade] = []

        # 每日历史记录
        self.daily_history: List[dict] = []

        # 总资产
        self.total_value = initial_capital

        # T+1 限制：当天买入的股票不能当天卖出
        self.buy_dates: Dict[str, date] = {}

    def update(self,
               current_date: date,
               signals: List[Signal],
               data_handler,
               transaction_cost_model) -> None:
        """
        根据信号更新投资组合

        Args:
            current_date: 当前日期
            signals: 交易信号列表
            data_handler: 数据处理器
            transaction_cost_model: 交易成本模型
        """
        # 先更新所有持仓的当前价格
        self._update_positions_price(data_handler, current_date)

        # 执行交易信号
        for signal in signals:
            self._execute_signal(signal, data_handler, transaction_cost_model)

        # 计算总资产
        self._calculate_total_value()

        # 记录每日状态
        self._record_daily_status(current_date)

    def _update_positions_price(self, data_handler, current_date: date) -> None:
        """更新所有持仓的当前价格"""
        for code, position in self.positions.items():
            if position.shares > 0:
                # 获取当日数据
                daily_data = data_handler.get_daily_data(current_date)
                if code in daily_data.index:
                    current_price = daily_data.loc[code, 'close']
                    position.update_price(current_price)

    def _execute_signal(self,
                        signal: Signal,
                        data_handler,
                        transaction_cost_model) -> None:
        """
        执行单个交易信号

        Args:
            signal: 交易信号
            data_handler: 数据处理器
            transaction_cost_model: 交易成本模型
        """
        if signal.signal_type == Signal.BUY:
            self._execute_buy(signal, data_handler, transaction_cost_model)
        elif signal.signal_type == Signal.SELL:
            self._execute_sell(signal, data_handler, transaction_cost_model)

    def _execute_buy(self,
                     signal: Signal,
                     data_handler,
                     transaction_cost_model) -> None:
        """执行买入"""
        code = signal.code
        price = signal.price

        # 计算目标金额
        target_amount = self.total_value * signal.weight

        # 检查单只股票最大仓位限制
        max_amount = self.total_value * self.max_single_position_ratio
        target_amount = min(target_amount, max_amount)

        # 计算交易成本
        commission = transaction_cost_model.calculate_buy_cost(target_amount)

        # 可用金额（扣除佣金）
        available_amount = min(self.cash, target_amount - commission)

        if available_amount <= 0:
            return

        # 计算买入股数（A股100股为一手）
        shares = int(available_amount / price / 100) * 100

        if shares == 0:
            return

        # 应用滑点
        actual_price = transaction_cost_model.apply_slippage(price, is_buy=True)
        actual_amount = actual_price * shares

        # 重新计算佣金
        commission = transaction_cost_model.calculate_buy_cost(actual_amount)

        # 检查资金是否充足
        total_cost = actual_amount + commission
        if total_cost > self.cash:
            # 资金不足，减少股数
            shares = int((self.cash - commission) / actual_price / 100) * 100
            if shares == 0:
                return
            actual_amount = actual_price * shares
            total_cost = actual_amount + commission

        # 执行买入
        if code not in self.positions:
            self.positions[code] = Position(code=code)

        self.positions[code].buy(shares, actual_price)
        # 立即更新持仓市值
        self.positions[code].update_price(actual_price)
        self.cash -= total_cost

        # 记录买入日期（T+1限制）
        self.buy_dates[code] = signal.date

        # 记录交易
        trade = Trade(
            date=signal.date,
            code=code,
            action='buy',
            shares=shares,
            price=actual_price,
            amount=actual_amount,
            commission=commission,
            reason=signal.reason
        )
        self.trades.append(trade)

    def _execute_sell(self,
                      signal: Signal,
                      data_handler,
                      transaction_cost_model) -> None:
        """执行卖出"""
        code = signal.code

        # 检查是否持仓
        if code not in self.positions or self.positions[code].shares == 0:
            return

        # T+1检查：当天买入不能当天卖出
        if code in self.buy_dates:
            if self.buy_dates[code] >= signal.date:
                return

        position = self.positions[code]
        price = signal.price

        # 应用滑点
        actual_price = transaction_cost_model.apply_slippage(price, is_buy=False)

        # 计算卖出股数（全部卖出）
        shares = position.shares

        # 计算金额
        amount = actual_price * shares

        # 计算交易成本
        commission = transaction_cost_model.calculate_sell_cost(amount)

        # 计算实际收入
        actual_income = amount - commission

        # 执行卖出
        realized_profit = position.sell(shares, actual_price)
        self.cash += actual_income

        # 清除买入日期记录
        if code in self.buy_dates:
            del self.buy_dates[code]

        # 记录交易
        trade = Trade(
            date=signal.date,
            code=code,
            action='sell',
            shares=shares,
            price=actual_price,
            amount=amount,
            commission=commission,
            reason=signal.reason
        )
        self.trades.append(trade)

    def _calculate_total_value(self) -> None:
        """计算总资产"""
        market_value = sum(pos.get_value() for pos in self.positions.values())
        self.total_value = self.cash + market_value

    def _record_daily_status(self, current_date: date) -> None:
        """记录每日状态"""
        self.daily_history.append({
            'date': current_date,
            'cash': self.cash,
            'market_value': self.total_value - self.cash,
            'total_value': self.total_value,
            'positions_count': len([pos for pos in self.positions.values()
                                   if pos.shares > 0])
        })

    def get_position(self, code: str) -> Optional[Position]:
        """
        获取指定股票的持仓

        Args:
            code: 股票代码

        Returns:
            持仓对象，如果没有持仓则返回None
        """
        return self.positions.get(code)
    
    def get_cash(self) -> float:
        """
        获取当前现金余额

        Returns:
            现金余额
        """
        return self.cash

    def get_all_positions(self) -> Dict[str, Position]:
        """
        获取所有持仓

        Returns:
            持仓字典
        """
        return {code: pos for code, pos in self.positions.items()
                if pos.shares > 0}

    def get_position_count(self) -> int:
        """
        获取持仓数量

        Returns:
            持仓数量
        """
        return len([pos for pos in self.positions.values() if pos.shares > 0])

    def get_position_value(self, data_handler, code: str, current_date: date) -> float:
        """
        获取单个持仓的市值

        Args:
            data_handler: 数据处理器
            code: 股票代码
            current_date: 当前日期

        Returns:
            持仓市值
        """
        if code not in self.positions:
            return 0.0

        position = self.positions[code]
        if position.shares == 0:
            return 0.0

        # 获取当前价格
        daily_data = data_handler.get_daily_data(current_date)
        if code not in daily_data.index:
            return position.market_value

        current_price = daily_data.loc[code, 'close']
        return position.shares * current_price

    def calculate_value(self, data_handler, current_date: date) -> float:
        """
        计算总资产价值

        Args:
            data_handler: 数据处理器
            current_date: 当前日期

        Returns:
            总资产价值
        """
        market_value = 0.0

        for code, position in self.positions.items():
            if position.shares > 0:
                market_value += self.get_position_value(
                    data_handler, code, current_date
                )

        return self.cash + market_value

    def get_daily_history(self) -> List[dict]:
        """
        获取每日历史记录

        Returns:
            每日历史记录列表
        """
        return self.daily_history

    def get_trades(self) -> List[Trade]:
        """
        获取所有交易记录

        Returns:
            交易记录列表
        """
        return self.trades

    def reset(self) -> None:
        """重置投资组合"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_history = []
        self.total_value = self.initial_capital
        self.buy_dates = {}

    def __repr__(self) -> str:
        return (f"Portfolio(total_value={self.total_value:.2f}, "
                f"cash={self.cash:.2f}, "
                f"positions={self.get_position_count()})")

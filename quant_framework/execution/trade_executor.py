"""
交易执行模块
负责订单的执行和管理
"""

from typing import List, Dict, Optional
from datetime import date
from dataclasses import dataclass

from ..strategy.base_strategy import Signal
from ..portfolio.portfolio import Portfolio
from ..execution.transaction_cost import TransactionCost
from ..utils.constraints import TradingConstraints


@dataclass
class Order:
    """订单类"""
    code: str
    action: str  # 'buy' or 'sell'
    shares: int
    price: float
    date: date
    status: str = "pending"  # pending, executed, cancelled, failed
    reason: str = ""

    def __repr__(self) -> str:
        return (f"Order(code={self.code}, action={self.action}, "
                f"shares={self.shares}, price={self.price:.2f}, "
                f"status={self.status})")


class TradeExecutor:
    """
    交易执行器
    负责将信号转换为订单并执行
    """

    def __init__(self,
                 transaction_cost: TransactionCost,
                 check_constraints: bool = True):
        """
        初始化交易执行器

        Args:
            transaction_cost: 交易成本模型
            check_constraints: 是否检查交易限制
        """
        self.transaction_cost = transaction_cost
        self.check_constraints = check_constraints
        self.orders: List[Order] = []
        self.execution_history: List[Order] = []

    def execute_signals(self,
                       signals: List[Signal],
                       portfolio: Portfolio,
                       data_handler,
                       current_date: date) -> List[Order]:
        """
        执行交易信号

        Args:
            signals: 交易信号列表
            portfolio: 投资组合
            data_handler: 数据处理器
            current_date: 当前日期

        Returns:
            已执行的订单列表
        """
        executed_orders = []

        for signal in signals:
            order = self._create_order_from_signal(signal)

            # 验证订单
            if not self._validate_order(order, signal, portfolio,
                                       data_handler, current_date):
                continue

            # 执行订单
            if self._execute_order(order, signal, portfolio):
                executed_orders.append(order)
                self.execution_history.append(order)

        return executed_orders

    def _create_order_from_signal(self, signal: Signal) -> Order:
        """从信号创建订单"""
        action = "buy" if signal.signal_type == Signal.BUY else "sell"
        return Order(
            code=signal.code,
            action=action,
            shares=0,  # 稍后计算
            price=signal.price,
            date=signal.date,
            reason=signal.reason
        )

    def _validate_order(self,
                       order: Order,
                       signal: Signal,
                       portfolio: Portfolio,
                       data_handler,
                       current_date: date) -> bool:
        """
        验证订单是否可以执行

        Args:
            order: 订单
            signal: 信号
            portfolio: 投资组合
            data_handler: 数据处理器
            current_date: 当前日期

        Returns:
            是否可以执行
        """
        # 检查交易限制
        if self.check_constraints:
            if not TradingConstraints.can_trade(
                order.code, current_date, data_handler, order.action
            ):
                order.status = "failed"
                return False

        # 检查资金是否充足（买入）
        if order.action == "buy":
            # 估算需要的资金
            estimated_cost = order.price * 100  # 至少一手
            commission = self.transaction_cost.calculate_buy_cost(estimated_cost)
            total_cost = estimated_cost + commission

            if portfolio.cash < total_cost:
                order.status = "failed"
                return False

        # 检查是否有持仓（卖出）
        elif order.action == "sell":
            position = portfolio.get_position(order.code)
            if not position or position.shares == 0:
                order.status = "failed"
                return False

            # T+1检查
            if order.code in portfolio.buy_dates:
                if portfolio.buy_dates[order.code] >= current_date:
                    order.status = "failed"
                    return False

        return True

    def _execute_order(self,
                      order: Order,
                      signal: Signal,
                      portfolio: Portfolio) -> bool:
        """
        执行订单

        Args:
            order: 订单
            signal: 信号
            portfolio: 投资组合

        Returns:
            是否执行成功
        """
        try:
            if order.action == "buy":
                success = self._execute_buy(order, signal, portfolio)
            else:
                success = self._execute_sell(order, signal, portfolio)

            order.status = "executed" if success else "failed"
            return success

        except Exception as e:
            print(f"执行订单失败: {e}")
            order.status = "failed"
            return False

    def _execute_buy(self,
                    order: Order,
                    signal: Signal,
                    portfolio: Portfolio) -> bool:
        """执行买入"""
        # 计算目标金额
        target_amount = portfolio.total_value * signal.weight

        # 单只股票最大仓位限制
        max_amount = portfolio.total_value * portfolio.max_single_position_ratio
        target_amount = min(target_amount, max_amount)

        # 计算佣金
        commission = self.transaction_cost.calculate_buy_cost(target_amount)

        # 可用金额
        available_amount = min(portfolio.cash, target_amount - commission)

        if available_amount <= 0:
            return False

        # 应用滑点
        actual_price = self.transaction_cost.apply_slippage(order.price, is_buy=True)

        # 计算股数（100股为一手）
        shares = int(available_amount / actual_price / 100) * 100

        if shares == 0:
            return False

        # 计算实际金额和佣金
        actual_amount = actual_price * shares
        actual_commission = self.transaction_cost.calculate_buy_cost(actual_amount)

        # 检查资金
        total_cost = actual_amount + actual_commission
        if total_cost > portfolio.cash:
            # 调整股数
            shares = int((portfolio.cash - actual_commission) / actual_price / 100) * 100
            if shares == 0:
                return False
            actual_amount = actual_price * shares
            total_cost = actual_amount + actual_commission

        # 更新订单
        order.shares = shares
        order.price = actual_price

        return True

    def _execute_sell(self,
                     order: Order,
                     signal: Signal,
                     portfolio: Portfolio) -> bool:
        """执行卖出"""
        position = portfolio.get_position(order.code)

        if not position or position.shares == 0:
            return False

        # 全部卖出
        shares = position.shares

        # 应用滑点
        actual_price = self.transaction_cost.apply_slippage(order.price, is_buy=False)

        # 更新订单
        order.shares = shares
        order.price = actual_price

        return True

    def get_execution_history(self) -> List[Order]:
        """获取执行历史"""
        return self.execution_history

    def clear_execution_history(self) -> None:
        """清除执行历史"""
        self.execution_history = []

    def __repr__(self) -> str:
        return f"TradeExecutor(executed_orders={len(self.execution_history)})"

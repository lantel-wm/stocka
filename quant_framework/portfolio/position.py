"""
持仓管理模块
管理单个持仓的信息
"""

from datetime import date
from typing import Optional
from dataclasses import dataclass


@dataclass
class Position:
    """
    持仓数据类
    记录单只股票的持仓信息
    """
    code: str  # 股票代码
    shares: int = 0  # 持仓股数
    avg_price: float = 0.0  # 平均成本价
    current_price: float = 0.0  # 当前价格
    market_value: float = 0.0  # 市值
    cost_value: float = 0.0  # 成本
    profit_loss: float = 0.0  # 盈亏金额
    profit_loss_ratio: float = 0.0  # 盈亏比例

    def __repr__(self) -> str:
        return (f"Position(code={self.code}, shares={self.shares}, "
                f"avg_price={self.avg_price:.2f}, "
                f"market_value={self.market_value:.2f}, "
                f"P&L={self.profit_loss:.2f})")

    def buy(self, shares: int, price: float) -> None:
        """
        买入股票

        Args:
            shares: 买入股数
            price: 买入价格
        """
        # 计算新的平均成本
        total_cost = self.cost_value + (shares * price)
        total_shares = self.shares + shares

        self.shares = total_shares
        self.avg_price = total_cost / total_shares if total_shares > 0 else 0.0
        self.current_price = price
        self.cost_value = total_cost

    def sell(self, shares: int, price: float) -> float:
        """
        卖出股票

        Args:
            shares: 卖出股数
            price: 卖出价格

        Returns:
            已实现盈亏
        """
        # 计算已实现盈亏
        realized_profit = (price - self.avg_price) * shares

        # 更新持仓
        self.shares -= shares
        self.current_price = price
        self.cost_value = self.shares * self.avg_price

        # 更新市值
        self.market_value = self.shares * price

        # 计算盈亏
        self.profit_loss = self.market_value - self.cost_value
        self.profit_loss_ratio = (self.profit_loss / self.cost_value
                                  if self.cost_value > 0 else 0.0)

        # 如果清仓，重置
        if self.shares == 0:
            self.avg_price = 0.0
            self.cost_value = 0.0
            self.market_value = 0.0

        return realized_profit

    def update_price(self, price: float) -> None:
        """
        更新当前价格

        Args:
            price: 当前价格
        """
        self.current_price = price
        self.market_value = self.shares * price
        self.profit_loss = self.market_value - self.cost_value
        self.profit_loss_ratio = (self.profit_loss / self.cost_value
                                  if self.cost_value > 0 else 0.0)

    def get_value(self) -> float:
        """
        获取持仓市值

        Returns:
            市值
        """
        return self.market_value

    def is_empty(self) -> bool:
        """
        是否空仓

        Returns:
            是否空仓
        """
        return self.shares == 0

    def to_dict(self) -> dict:
        """
        转换为字典

        Returns:
            字典格式的持仓信息
        """
        return {
            'code': self.code,
            'shares': self.shares,
            'avg_price': self.avg_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'cost_value': self.cost_value,
            'profit_loss': self.profit_loss,
            'profit_loss_ratio': self.profit_loss_ratio
        }

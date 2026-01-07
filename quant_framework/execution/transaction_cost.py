"""
交易成本计算模块
实现A股标准交易成本计算（佣金、印花税、滑点）
"""

from typing import Optional


class TransactionCost:
    """
    交易成本计算类
    按照A股标准计算交易成本
    """

    def __init__(self,
                 commission_rate: float = 0.0003,
                 stamp_duty_rate: float = 0.001,
                 min_commission: float = 5.0,
                 slippage: float = 0.001):
        """
        初始化交易成本模型

        Args:
            commission_rate: 佣金率（默认万三，即0.03%）
            stamp_duty_rate: 印花税率（仅卖出收取，默认千分之一）
            min_commission: 最低佣金（默认5元）
            slippage: 滑点（默认0.1%）

        A股交易成本说明：
        - 佣金：双向收取，最低5元，费率一般在万一到万三之间
        - 印花税：仅卖出收取，费率为千分之一
        - 过户费：双向收取，费率为万分之一（已包含在佣金中）
        """
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_commission = min_commission
        self.slippage = slippage

    def calculate_buy_cost(self, amount: float) -> float:
        """
        计算买入成本

        买入时：佣金（最低5元）

        Args:
            amount: 交易金额

        Returns:
            买入成本（佣金）
        """
        commission = max(amount * self.commission_rate, self.min_commission)
        return commission

    def calculate_sell_cost(self, amount: float) -> float:
        """
        计算卖出成本

        卖出时：佣金（最低5元）+ 印花税（千分之一）

        Args:
            amount: 交易金额

        Returns:
            卖出成本（佣金 + 印花税）
        """
        # 佣金
        commission = max(amount * self.commission_rate, self.min_commission)

        # 印花税（仅卖出）
        stamp_duty = amount * self.stamp_duty_rate

        return commission + stamp_duty

    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        应用滑点

        买入时价格向上滑点（不利），卖出时价格向下滑点（不利）

        Args:
            price: 原始价格
            is_buy: 是否为买入

        Returns:
            应用滑点后的价格
        """
        if is_buy:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def calculate_total_cost(self,
                            price: float,
                            shares: int,
                            is_buy: bool) -> float:
        """
        计算总交易成本

        Args:
            price: 交易价格
            shares: 交易股数（A股以100股为一手，必须是100的整数倍）
            is_buy: 是否为买入

        Returns:
            总成本（包括交易费用 + 滑点损失）
        """
        amount = price * shares

        # 应用滑点
        actual_price = self.apply_slippage(price, is_buy)
        actual_amount = actual_price * shares

        # 计算交易费用
        if is_buy:
            cost = self.calculate_buy_cost(actual_amount)
        else:
            cost = self.calculate_sell_cost(actual_amount)

        # 返回总成本（交易费用 + 滑点损失）
        slippage_cost = abs(actual_amount - amount)
        return cost + slippage_cost

    def calculate_commission_only(self, amount: float) -> float:
        """
        仅计算佣金（不含印花税）

        Args:
            amount: 交易金额

        Returns:
            佣金金额
        """
        return max(amount * self.commission_rate, self.min_commission)

    def estimate_total_cost_percent(self, is_buy: bool = True) -> float:
        """
        估算总成本百分比（佣金 + 印花税 + 滑点）

        Args:
            is_buy: 是否为买入

        Returns:
            总成本百分比
        """
        if is_buy:
            # 买入：佣金 + 滑点
            return self.commission_rate + self.slippage
        else:
            # 卖出：佣金 + 印花税 + 滑点
            return self.commission_rate + self.stamp_duty_rate + self.slippage

    def __repr__(self) -> str:
        return (f"TransactionCost(commission_rate={self.commission_rate:.4f}, "
                f"stamp_duty_rate={self.stamp_duty_rate:.4f}, "
                f"min_commission={self.min_commission:.2f}, "
                f"slippage={self.slippage:.4f})")


# 预定义的交易成本模型

class StandardCost(TransactionCost):
    """A股标准交易成本（万三佣金 + 千一印花税 + 0.1%滑点）"""

    def __init__(self):
        super().__init__(
            commission_rate=0.0003,  # 万三
            stamp_duty_rate=0.001,   # 千一
            min_commission=5.0,
            slippage=0.001
        )


class LowCost(TransactionCost):
    """低交易成本（万一佣金 + 千一印花税 + 0.05%滑点）"""

    def __init__(self):
        super().__init__(
            commission_rate=0.0001,  # 万一
            stamp_duty_rate=0.001,   # 千一
            min_commission=5.0,
            slippage=0.0005
        )


class ZeroCost(TransactionCost):
    """零成本（用于测试）"""

    def __init__(self):
        super().__init__(
            commission_rate=0.0,
            stamp_duty_rate=0.0,
            min_commission=0.0,
            slippage=0.0
        )

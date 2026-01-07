"""
交易成本测试
测试交易成本计算的正确性
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant_framework.execution.transaction_cost import (
    TransactionCost,
    StandardCost,
    LowCost,
    ZeroCost
)


class TestTransactionCost:
    """交易成本测试类"""

    def test_buy_cost_calculation(self):
        """测试买入成本计算"""
        cost_model = TransactionCost()

        # 测试正常买入
        amount = 10000.0
        cost = cost_model.calculate_buy_cost(amount)

        expected_commission = max(amount * 0.0003, 5.0)
        assert cost == expected_commission

    def test_sell_cost_calculation(self):
        """测试卖出成本计算"""
        cost_model = TransactionCost()

        # 测试正常卖出
        amount = 10000.0
        cost = cost_model.calculate_sell_cost(amount)

        expected_commission = max(amount * 0.0003, 5.0)
        expected_stamp_duty = amount * 0.001
        expected_total = expected_commission + expected_stamp_duty

        assert cost == expected_total

    def test_minimum_commission(self):
        """测试最低佣金"""
        cost_model = TransactionCost()

        # 小额交易，应该收取最低佣金5元
        small_amount = 1000.0
        cost = cost_model.calculate_buy_cost(small_amount)

        assert cost == 5.0

    def test_slippage_buy(self):
        """测试买入滑点"""
        cost_model = TransactionCost(slippage=0.001)

        price = 10.0
        adjusted_price = cost_model.apply_slippage(price, is_buy=True)

        expected_price = price * 1.001
        assert abs(adjusted_price - expected_price) < 0.0001

    def test_slippage_sell(self):
        """测试卖出滑点"""
        cost_model = TransactionCost(slippage=0.001)

        price = 10.0
        adjusted_price = cost_model.apply_slippage(price, is_buy=False)

        expected_price = price * 0.999
        assert abs(adjusted_price - expected_price) < 0.0001

    def test_total_cost(self):
        """测试总成本计算"""
        cost_model = TransactionCost()

        price = 10.0
        shares = 1000
        is_buy = True

        total_cost = cost_model.calculate_total_cost(price, shares, is_buy)

        assert total_cost > 0
        # 总成本应该包含佣金和滑点损失
        assert total_cost > (price * shares * cost_model.commission_rate)

    def test_standard_cost(self):
        """测试标准交易成本"""
        cost_model = StandardCost()

        assert cost_model.commission_rate == 0.0003
        assert cost_model.stamp_duty_rate == 0.001
        assert cost_model.min_commission == 5.0
        assert cost_model.slippage == 0.001

    def test_low_cost(self):
        """测试低交易成本"""
        cost_model = LowCost()

        assert cost_model.commission_rate == 0.0001
        assert cost_model.stamp_duty_rate == 0.001
        assert cost_model.slippage == 0.0005

    def test_zero_cost(self):
        """测试零成本"""
        cost_model = ZeroCost()

        assert cost_model.commission_rate == 0.0
        assert cost_model.stamp_duty_rate == 0.0
        assert cost_model.min_commission == 0.0
        assert cost_model.slippage == 0.0

        # 零成本应该返回0
        amount = 10000.0
        buy_cost = cost_model.calculate_buy_cost(amount)
        sell_cost = cost_model.calculate_sell_cost(amount)

        assert buy_cost == 0.0
        assert sell_cost == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

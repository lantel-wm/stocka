"""
测试因子相关功能

运行:
    pytest tests/test_factor_handler.py -v
"""

import unittest
import tempfile
import os
from datetime import date
import pandas as pd
import numpy as np

from quant_framework.data.data_handler import DataHandler


class TestFactorHandler(unittest.TestCase):
    """测试因子存储和查询功能"""

    @classmethod
    def setUpClass(cls):
        """测试前准备：创建临时数据库"""
        # 创建临时数据库文件
        cls.db_fd, cls.db_path = tempfile.mkstemp(suffix='.db')

        # 创建一个简单的测试数据库（包含股票行情表）
        import duckdb
        con = duckdb.connect(cls.db_path)

        # 创建股票行情表
        con.execute("""
            CREATE TABLE stock_prices (
                code VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DOUBLE,
                close DOUBLE,
                high DOUBLE,
                low DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (code, date)
            )
        """)

        # 插入测试数据
        test_data = []
        codes = ['000001', '000002', '600000']
        dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]

        for code in codes:
            for d in dates:
                test_data.append({
                    'code': code,
                    'date': d,
                    'open': 10.0 + np.random.rand(),
                    'close': 10.0 + np.random.rand(),
                    'high': 11.0 + np.random.rand(),
                    'low': 9.0 + np.random.rand(),
                    'volume': 1000000 + np.random.randint(0, 100000)
                })

        df = pd.DataFrame(test_data)
        con.register('temp_stock', df)
        con.execute("INSERT INTO stock_prices SELECT * FROM temp_stock")
        con.unregister('temp_stock')
        con.close()

    @classmethod
    def tearDownClass(cls):
        """测试后清理：删除临时数据库"""
        os.close(cls.db_fd)
        os.unlink(cls.db_path)

    def setUp(self):
        """每个测试前的准备"""
        self.handler = DataHandler(self.db_path, table_name='stock_prices')

    def tearDown(self):
        """每个测试后的清理"""
        self.handler.close()

    def test_init_factor_tables(self):
        """测试因子表初始化"""
        # 初始化因子表
        self.handler._init_factor_tables()

        # 检查表是否创建
        tables = self.handler.con.execute("SHOW TABLES").fetchdf()
        table_names = tables['name'].tolist()

        self.assertIn('factor_definitions', table_names)
        self.assertIn('factor_data', table_names)

    def test_register_factor(self):
        """测试因子注册"""
        self.handler._init_factor_tables()

        # 注册因子（指定ID）
        factor_id = self.handler.register_factor(1, 'TEST_FACTOR', 'custom', '测试因子')

        self.assertIsNotNone(factor_id)
        self.assertEqual(factor_id, 1)

        # 重复注册应返回相同的ID
        factor_id2 = self.handler.register_factor(1, 'TEST_FACTOR', 'custom', '测试因子')
        self.assertEqual(factor_id, factor_id2)

        # 测试冲突处理
        with self.assertRaises(ValueError):
            # 尝试用不同名称注册已存在的ID
            self.handler.register_factor(1, 'OTHER_FACTOR', 'custom', '其他因子')

    def test_get_factor_id(self):
        """测试获取因子ID"""
        self.handler._init_factor_tables()

        # 注册因子（指定ID）
        self.handler.register_factor(18, 'MA5', 'alpha158', '5日均线')

        # 获取因子ID
        factor_id = self.handler.get_factor_id('MA5')
        self.assertIsNotNone(factor_id)
        self.assertEqual(factor_id, 18)

        # 获取不存在的因子
        factor_id_none = self.handler.get_factor_id('NOT_EXIST')
        self.assertIsNone(factor_id_none)

    def test_get_available_factors(self):
        """测试获取可用因子列表"""
        self.handler._init_factor_tables()

        # 注册多个因子（指定ID）
        self.handler.register_factor(18, 'MA5', 'alpha158', '5日均线')
        self.handler.register_factor(19, 'MA10', 'alpha158', '10日均线')
        self.handler.register_factor(1000, 'CUSTOM_FACTOR', 'custom', '自定义因子')

        # 获取因子列表
        factors = self.handler.get_available_factors()

        self.assertEqual(len(factors), 3)
        self.assertIn('MA5', factors)
        self.assertIn('MA10', factors)
        self.assertIn('CUSTOM_FACTOR', factors)

    def test_save_and_get_factors(self):
        """测试保存和查询因子值"""
        self.handler._init_factor_tables()

        # 注册因子（指定ID）
        self.handler.register_factor(18, 'MA5', 'alpha158', '5日均线')
        self.handler.register_factor(19, 'MA10', 'alpha158', '10日均线')

        # 准备测试数据
        factor_df = pd.DataFrame({
            'MA5': [1.02, 0.98, 1.01],
            'MA10': [0.99, 1.02, 0.97]
        }, index=['000001', '000002', '600000'])

        # 保存因子
        self.handler.save_factors(factor_df, date(2024, 1, 2))

        # 查询截面因子
        cross_section = self.handler.get_factor_cross_section(
            date(2024, 1, 2),
            factor_names=['MA5', 'MA10']
        )

        self.assertEqual(len(cross_section), 3)
        self.assertIn('MA5', cross_section.columns)
        self.assertIn('MA10', cross_section.columns)

        # 检查具体值
        self.assertAlmostEqual(cross_section.loc['000001', 'MA5'], 1.02, places=5)

    def test_get_stock_factors(self):
        """测试获取单只股票的时序因子"""
        self.handler._init_factor_tables()

        # 注册因子（指定ID）
        self.handler.register_factor(18, 'MA5', 'alpha158', '5日均线')

        # 保存多个日期的数据
        factor_df1 = pd.DataFrame({
            'MA5': [1.02, 0.98, 1.01]
        }, index=['000001', '000002', '600000'])

        factor_df2 = pd.DataFrame({
            'MA5': [1.03, 0.97, 1.00]
        }, index=['000001', '000002', '600000'])

        self.handler.save_factors(factor_df1, date(2024, 1, 1))
        self.handler.save_factors(factor_df2, date(2024, 1, 2))

        # 查询单只股票的时序数据
        stock_factors = self.handler.get_stock_factors(
            '000001',
            factor_names=['MA5'],
            target_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2)
        )

        self.assertEqual(len(stock_factors), 2)
        self.assertIn('MA5', stock_factors.columns)
        self.assertAlmostEqual(stock_factors.loc[date(2024, 1, 1), 'MA5'], 1.02, places=5)
        self.assertAlmostEqual(stock_factors.loc[date(2024, 1, 2), 'MA5'], 1.03, places=5)

    def test_get_factors_wide(self):
        """测试获取宽表格式的因子数据"""
        self.handler._init_factor_tables()

        # 注册因子（指定ID）
        self.handler.register_factor(18, 'MA5', 'alpha158', '5日均线')
        self.handler.register_factor(19, 'MA10', 'alpha158', '10日均线')

        # 保存因子
        factor_df = pd.DataFrame({
            'MA5': [1.02, 0.98, 1.01],
            'MA10': [0.99, 1.02, 0.97]
        }, index=['000001', '000002', '600000'])

        self.handler.save_factors(factor_df, date(2024, 1, 2))

        # 查询宽表格式
        wide_df = self.handler.get_factors_wide(
            trade_date=date(2024, 1, 2),
            stock_codes=['000001', '000002'],
            factor_names=['MA5', 'MA10']
        )

        self.assertEqual(len(wide_df), 2)
        self.assertIn('MA5', wide_df.columns)
        self.assertIn('MA10', wide_df.columns)

        # 检查索引
        self.assertIn('000001', wide_df.index)
        self.assertIn('000002', wide_df.index)

    def test_get_factor_info(self):
        """测试获取因子定义信息"""
        self.handler._init_factor_tables()

        # 注册因子（指定ID）
        self.handler.register_factor(18, 'MA5', 'alpha158', '5日均线')
        self.handler.register_factor(1000, 'CUSTOM_FACTOR', 'custom', '自定义因子')

        # 获取因子信息
        factor_info = self.handler.get_factor_info()

        self.assertEqual(len(factor_info), 2)
        self.assertIn('factor_id', factor_info.columns)
        self.assertIn('factor_name', factor_info.columns)
        self.assertIn('factor_category', factor_info.columns)

    def test_save_factors_with_nan(self):
        """测试保存包含NaN的因子数据"""
        self.handler._init_factor_tables()

        # 注册因子（指定ID）
        self.handler.register_factor(18, 'MA5', 'alpha158', '5日均线')

        # 准备包含NaN的数据
        factor_df = pd.DataFrame({
            'MA5': [1.02, np.nan, 1.01]
        }, index=['000001', '000002', '600000'])

        # 保存因子（应该跳过NaN值）
        self.handler.save_factors(factor_df, date(2024, 1, 2))

        # 查询并验证
        cross_section = self.handler.get_factor_cross_section(
            date(2024, 1, 2),
            factor_names=['MA5']
        )

        # 000002应该不在结果中（因为其值为NaN）
        self.assertEqual(len(cross_section), 2)
        self.assertNotIn('000002', cross_section.index)


if __name__ == '__main__':
    unittest.main()

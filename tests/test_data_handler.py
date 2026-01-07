"""
数据处理器测试
测试数据加载和查询功能
"""

import sys
import os
import pytest
import pandas as pd
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant_framework.data.data_handler import DataHandler


class TestDataHandler:
    """数据处理器测试类"""

    @pytest.fixture
    def sample_data_path(self, tmp_path):
        """创建测试数据目录"""
        # 创建临时测试数据
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        # 创建测试CSV文件
        test_data = pd.DataFrame({
            '日期': ['2020-01-01', '2020-01-02', '2020-01-03'],
            '股票代码': ['000001', '000001', '000001'],
            '开盘': [10.0, 10.5, 11.0],
            '收盘': [10.5, 11.0, 11.5],
            '最高': [10.8, 11.2, 11.8],
            '最低': [9.8, 10.3, 10.8],
            '成交量': [1000000, 1200000, 900000],
            '成交额': [10500000, 13200000, 10350000]
        })

        test_file = data_dir / "000001.csv"
        test_data.to_csv(test_file, index=False, encoding='utf-8')

        return str(data_dir)

    def test_data_handler_initialization(self, sample_data_path):
        """测试数据处理器初始化"""
        handler = DataHandler(sample_data_path)

        assert handler.data_path == sample_data_path
        assert handler.all_data is None
        assert handler.stock_codes == []

    def test_load_data(self, sample_data_path):
        """测试数据加载"""
        handler = DataHandler(sample_data_path)
        handler.load_data()

        assert handler.all_data is not None
        assert len(handler.stock_codes) > 0
        assert '000001' in handler.stock_codes

    def test_get_stock_data(self, sample_data_path):
        """测试获取单只股票数据"""
        handler = DataHandler(sample_data_path)
        handler.load_data()

        df = handler.get_stock_data('000001')

        assert df is not None
        assert len(df) > 0
        assert 'close' in df.columns

    def test_get_available_dates(self, sample_data_path):
        """测试获取可用日期"""
        handler = DataHandler(sample_data_path)
        handler.load_data()

        dates = handler.get_available_dates()

        assert len(dates) > 0
        assert isinstance(dates[0], date)

    def test_data_info(self, sample_data_path):
        """测试数据信息"""
        handler = DataHandler(sample_data_path)
        handler.load_data()

        info = handler.get_data_info()

        assert info['status'] == "已加载"
        assert info['stock_count'] > 0
        assert 'start_date' in info
        assert 'end_date' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

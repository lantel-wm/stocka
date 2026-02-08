"""
DataUpdater 使用示例

演示如何使用 DataUpdater 从 akshare 获取最新的 A 股行情数据并更新到数据库。

功能：
- 更新单只股票数据
- 批量更新多只股票数据
- 处理更新结果和错误
- 展示增量更新机制
"""

import os
import sys

# 添加项目根目录到 Python 路径
STOCKA_BASE_DIR = '/home/zzy/projects/stocka'
sys.path.insert(0, str(STOCKA_BASE_DIR))

from quant_framework.data.data_handler import DataHandler
from quant_framework.realtime.data_updater import DataUpdater
from quant_framework.utils.logger import get_logger

logger = get_logger(__name__)


def example_update_single_stock():
    """示例 1: 更新单只股票数据"""
    print("\n" + "=" * 60)
    print("示例 1: 更新单只股票数据")
    print("=" * 60)

    # 初始化 DataHandler
    db_path = os.path.join(STOCKA_BASE_DIR, "data/stock.db")
    handler = DataHandler(db_path)

    # 初始化 DataUpdater
    updater = DataUpdater(handler, delay=0.5)

    # 更新单只股票（浦发银行 600000）
    stock_code = "600000"
    print(f"\n正在更新股票 {stock_code}...")

    result = updater.update_stock_data(stock_code)

    # 展示更新结果
    print(f"\n更新结果:")
    print(f"  股票代码: {result['stock_code']}")
    print(f"  状态: {result['status']}")
    print(f"  消息: {result['message']}")
    print(f"  现有数据范围: {result['existing_range']}")
    print(f"  下载的数据范围: {result['downloaded_range']}")
    print(f"  新增行数: {result['new_rows']}")
    print(f"  总行数: {result['total_rows']}")

    # 验证数据是否已更新到数据库
    latest_date = handler.get_stock_latest_date(stock_code)
    print(f"\n验证: 数据库中最新日期范围: {latest_date}")


def example_update_batch_stocks():
    """示例 2: 批量更新多只股票数据"""
    print("\n" + "=" * 60)
    print("示例 2: 批量更新多只股票数据")
    print("=" * 60)

    # 初始化 DataHandler
    db_path = os.path.join(STOCKA_BASE_DIR, "data/stock.db")
    handler = DataHandler(db_path)

    # 初始化 DataUpdater
    updater = DataUpdater(handler, delay=0.5)

    # 定义要更新的股票列表
    stock_codes = [
        "600000",  # 浦发银行
        "600036",  # 招商银行
        "600519",  # 贵州茅台
        "000001",  # 平安银行
        "000002"   # 万科A
    ]

    print(f"\n正在批量更新 {len(stock_codes)} 只股票...")
    print(f"股票列表: {', '.join(stock_codes)}")

    # 批量更新
    results = updater.update_batch_stock_data(stock_codes)

    # 展示批量更新结果
    print(f"\n批量更新汇总:")
    print(f"  总计: {results['total']} 只")
    print(f"  成功: {results['success']} 只")
    print(f"  跳过: {results['skipped']} 只")
    print(f"  失败: {results['error']} 只")

    # 展示每只股票的详细结果
    print(f"\n详细信息:")
    for i, detail in enumerate(results['details'], 1):
        status_icon = {
            'success': '✓',
            'skipped': '⊘',
            'error': '✗'
        }.get(detail['status'], '?')

        print(f"  {i}. {status_icon} {detail['stock_code']}: {detail['message']}")
        if detail['status'] == 'success':
            print(f"     下载范围: {detail['downloaded_range']}, 新增: {detail['new_rows']} 条")


def example_incremental_update():
    """示例 3: 演示增量更新机制"""
    print("\n" + "=" * 60)
    print("示例 3: 演示增量更新机制")
    print("=" * 60)

    # 初始化 DataHandler
    db_path = os.path.join(STOCKA_BASE_DIR, "data/stock.db")
    handler = DataHandler(db_path)
    
    # 初始化 DataUpdater
    updater = DataUpdater(handler, delay=0.5)

    stock_code = "600000"

    # 第一次更新：获取当前数据范围
    print(f"\n第一次更新股票 {stock_code}...")
    result1 = updater.update_stock_data(stock_code)
    print(f"  现有数据范围: {result1['existing_range']}")
    print(f"  下载的数据范围: {result1['downloaded_range']}")
    print(f"  新增行数: {result1['new_rows']}")

    # 第二次更新：演示增量更新（应该跳过或下载很少数据）
    print(f"\n第二次更新同一股票 {stock_code}（演示增量更新）...")
    result2 = updater.update_stock_data(stock_code)
    print(f"  现有数据范围: {result2['existing_range']}")
    print(f"  下载的数据范围: {result2['downloaded_range']}")
    print(f"  新增行数: {result2['new_rows']}")
    print(f"  状态: {result2['status']}")


def example_custom_end_date():
    """示例 4: 指定结束日期更新"""
    print("\n" + "=" * 60)
    print("示例 4: 指定结束日期更新")
    print("=" * 60)

    # 初始化 DataHandler
    db_path = os.path.join(STOCKA_BASE_DIR, "data/stock.db")
    handler = DataHandler(db_path)

    # 初始化 DataUpdater
    updater = DataUpdater(handler, delay=0.5)

    stock_code = "600000"
    end_date = "20241201"  # 更新到 2024-12-01

    print(f"\n更新股票 {stock_code} 到指定日期 {end_date}...")

    result = updater.update_stock_data(stock_code, end_date=end_date)

    print(f"\n更新结果:")
    print(f"  状态: {result['status']}")
    print(f"  消息: {result['message']}")
    print(f"  下载的数据范围: {result['downloaded_range']}")


def example_error_handling():
    """示例 5: 错误处理"""
    print("\n" + "=" * 60)
    print("示例 5: 错误处理")
    print("=" * 60)

    # 初始化 DataHandler
    db_path = os.path.join(STOCKA_BASE_DIR, "data/stock.db")
    handler = DataHandler(db_path)

    # 初始化 DataUpdater
    updater = DataUpdater(handler, delay=0.5)

    # 测试不存在的股票代码
    invalid_code = "999999"
    print(f"\n尝试更新不存在的股票 {invalid_code}...")

    result = updater.update_stock_data(invalid_code)

    if result['status'] == 'error':
        print(f"✓ 正确处理了错误:")
        print(f"  错误消息: {result['message']}")
    else:
        print(f"✗ 预期应该返回错误状态")


def example_get_database_info():
    """示例 6: 查看数据库信息"""
    print("\n" + "=" * 60)
    print("示例 6: 查看数据库信息")
    print("=" * 60)

    # 初始化 DataHandler
    db_path = os.path.join(STOCKA_BASE_DIR, "data/stock.db")
    handler = DataHandler(db_path)

    # 获取数据库信息
    info = handler.get_data_info()

    print(f"\n数据库信息:")
    print(f"  数据库文件: {db_path}")
    print(f"  状态: {info['status']}")
    print(f"  股票数量: {info['stock_count']}")
    print(f"  数据起始日期: {info['start_date']}")
    print(f"  数据截止日期: {info['end_date']}")
    print(f"  交易日数量: {info['trading_days']}")
    print(f"  总数据点数: {info['total_data_points']}")

    # 获取所有股票代码
    all_codes = handler.get_all_codes()
    print(f"\n前 10 只股票代码: {', '.join(all_codes[:10])}")
    print(f"  总计: {len(all_codes)} 只股票")


def main():
    """主函数：运行所有示例"""
    print("\n" + "=" * 60)
    print("DataUpdater 使用示例")
    print("=" * 60)

    try:
        # 示例 6: 查看数据库信息（先运行这个，了解当前数据状态）
        # example_get_database_info()

        # 示例 1: 更新单只股票
        # example_update_single_stock()

        # 示例 2: 批量更新
        example_update_batch_stocks()

        # 示例 3: 增量更新
        # example_incremental_update()

        # 示例 4: 指定结束日期
        # example_custom_end_date()

        # 示例 5: 错误处理
        # example_error_handling()

        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)

        # 提示
        print("\n提示:")
        print("  1. 取消注释上面的示例函数以运行特定示例")
        print("  2. 确保已安装 akshare: pip install akshare")
        print("  3. 数据库文件路径: data/stock_data.duckdb")
        print("  4. 注意网络请求延迟，避免频繁调用")

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保数据库文件存在: data/stock_data.duckdb")
        print("您可以先运行数据初始化脚本创建数据库。")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
初始化股票信息表
从 data/stock/list 目录中的文件导入股票信息到 DuckDB
"""

import pandas as pd
import duckdb
from pathlib import Path

# 定义数据库路径
DB_PATH = "data/stock.db"
DATA_DIR = Path("data/stock/list")

def create_stock_info_table():
    """创建股票信息表并导入数据"""

    # 连接数据库
    conn = duckdb.connect(DB_PATH)

    try:
        # 创建 stock_info 表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_info (
                code VARCHAR(6) PRIMARY KEY,  -- 股票代码
                stock_name VARCHAR,            -- 股票简称
                stock_full_name VARCHAR,       -- 股票全称
                company_short_name VARCHAR,    -- 公司简称
                company_full_name VARCHAR,     -- 公司全称
                market VARCHAR,                -- 市场（上海/深圳）
                board VARCHAR,                 -- 板块（主板/创业板/科创板等）
                list_date DATE,                -- 上市日期
                total_shares BIGINT,           -- 总股本
                float_shares BIGINT,           -- 流通股本
                industry VARCHAR               -- 所属行业
            )
        """)
        print("✓ 创建 stock_info 表成功")

        # 读取上海股票列表
        sh_file = DATA_DIR / "sh_stock_list.csv"
        if sh_file.exists():
            df_sh = pd.read_csv(sh_file, encoding='utf-8')
            # 添加市场信息
            df_sh['market'] = '上海'
            df_sh['board'] = '主板'  # 上海主板
            # 重命名列
            df_sh.rename(columns={
                '证券代码': 'code',
                '证券简称': 'stock_name',
                '证券全称': 'stock_full_name',
                '公司简称': 'company_short_name',
                '公司全称': 'company_full_name',
                '上市日期': 'list_date'
            }, inplace=True)
            # 添加缺失的列
            df_sh['total_shares'] = None
            df_sh['float_shares'] = None
            df_sh['industry'] = None
            print(f"✓ 读取上海股票: {len(df_sh)} 只")
        else:
            df_sh = pd.DataFrame()
            print("⚠ 未找到上海股票文件")

        # 读取深圳股票列表
        sz_file = DATA_DIR / "sz_stock_list.csv"
        if sz_file.exists():
            df_sz = pd.read_csv(sz_file, encoding='utf-8')
            # 处理股本数据（去除逗号）
            if 'A股总股本' in df_sz.columns:
                df_sz['A股总股本'] = df_sz['A股总股本'].str.replace(',', '').astype(float)
            if 'A股流通股本' in df_sz.columns:
                df_sz['A股流通股本'] = df_sz['A股流通股本'].str.replace(',', '').astype(float)
            # 添加市场信息
            df_sz['market'] = '深圳'
            # 重命名列
            df_sz.rename(columns={
                'A股代码': 'code',
                'A股简称': 'stock_name',
                '板块': 'board',
                'A股上市日期': 'list_date',
                'A股总股本': 'total_shares',
                'A股流通股本': 'float_shares',
                '所属行业': 'industry'
            }, inplace=True)
            # 添加缺失的列
            df_sz['stock_full_name'] = None
            df_sz['company_short_name'] = None
            df_sz['company_full_name'] = None
            print(f"✓ 读取深圳股票: {len(df_sz)} 只")
        else:
            df_sz = pd.DataFrame()
            print("⚠ 未找到深圳股票文件")

        # 合并数据
        columns = ['code', 'stock_name', 'stock_full_name', 'company_short_name',
                   'company_full_name', 'market', 'board', 'list_date', 'total_shares',
                   'float_shares', 'industry']

        df_all = pd.concat([df_sh[columns], df_sz[columns]], ignore_index=True)
        print(f"✓ 合并后总计: {len(df_all)} 只股票")

        # 清空旧数据（如果存在）
        conn.execute("DELETE FROM stock_info")

        # 导入数据到数据库
        conn.register('df_stock', df_all)
        conn.execute("""
            INSERT INTO stock_info
            SELECT * FROM df_stock
        """)

        # 显示统计信息
        result = conn.execute("""
            SELECT
                market,
                board,
                COUNT(*) as count
            FROM stock_info
            GROUP BY market, board
            ORDER BY market, board
        """).fetchdf()

        print("\n股票信息统计:")
        print(result.to_string(index=False))

        # 显示示例数据
        sample = conn.execute("""
            SELECT code, stock_name, market, board, industry
            FROM stock_info
            LIMIT 10
        """).fetchdf()

        print("\n示例数据:")
        print(sample.to_string(index=False))

        print(f"\n✓ 成功导入 {len(df_all)} 条股票信息到 stock_info 表")

    except Exception as e:
        print(f"✗ 错误: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    create_stock_info_table()

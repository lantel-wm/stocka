"""
数据管理模块
负责加载、清洗和管理A股日线数据
基于DuckDB数据库，提供高性能查询接口
"""

import os
from datetime import date
from typing import Optional, List, Dict
import pandas as pd
import duckdb


class DataHandler:
    """
    数据处理器类
    基于DuckDB数据库加载A股日线数据，提供数据查询接口
    提供高性能查询和复杂SQL支持

    注意: 初始化后即可直接使用，无需调用load_data()
    """

    # 表名常量
    TABLE_PRICES = 'stock_prices'
    TABLE_STOCK_INFO = 'stock_info'
    TABLE_TRADING_DAYS = 'trading_days'
    TABLE_FACTOR_DEFINITIONS = 'factor_definitions'
    TABLE_FACTOR_DATA = 'factor_data'

    def __init__(self, db_path: str):
        """
        初始化数据处理器

        Args:
            db_path: DuckDB数据库文件路径
        """
        self.db_path = db_path
        self.available_dates = []
        self.stock_codes = []

        # 检查数据库文件是否存在
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")

        self.con = duckdb.connect(db_path)

        # 检查价格表是否存在
        table_exists = self.con.execute(f"""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = '{self.TABLE_PRICES}'
        """).fetchone()

        if not table_exists:
            raise ValueError(f"表不存在: {self.TABLE_PRICES}")

        # 自动加载元数据
        self._load_metadata()
        
        self._init_factor_tables()

    def _load_metadata(self) -> None:
        """加载元数据(股票代码列表、可用日期等)"""
        # 获取所有股票代码
        result = self.con.execute(f"""
            SELECT DISTINCT code FROM {self.TABLE_PRICES} ORDER BY code
        """).fetchdf()
        self.stock_codes = result['code'].tolist()

        # 获取所有可用交易日期
        result = self.con.execute(f"""
            SELECT DISTINCT date FROM {self.TABLE_PRICES} ORDER BY date
        """).fetchdf()
        self.available_dates = result['date'].tolist()
        # 转换为date对象
        self.available_dates = [d.date() if hasattr(d, 'date') else d
                               for d in self.available_dates]

    def get_stock_data(self, code: str,
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> pd.DataFrame:
        """
        获取单只股票的历史数据

        Args:
            code: 股票代码
            start_date: 开始日期（默认返回所有数据）
            end_date: 截止日期（默认返回所有数据）

        Returns:
            股票历史数据DataFrame
        """
        query = f"SELECT * FROM {self.TABLE_PRICES} WHERE code = ?"
        params = [str(code)]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        return self.con.execute(query, params).df()

    def get_stock_latest_date(self, code: str) -> tuple:
        """
        获取单只股票数据的日期范围

        Args:
            code: 股票代码

        Returns:
            (start_date, end_date) 日期元组，格式为 date 对象，如果无数据返回 (None, None)

        Example:
            >>> start, end = handler.get_stock_latest_date('600000')
            >>> print(f"数据范围: {start} 至 {end}")
        """
        try:
            result = self.con.execute(f"""
                SELECT
                    MIN(date) as start_date,
                    MAX(date) as end_date
                FROM {self.TABLE_PRICES}
                WHERE code = ?
            """, [str(code)]).fetchone()

            if result and result[0] is not None:
                return (result[0], result[1])
            return (None, None)

        except Exception:
            return (None, None)

    def get_data_before(self, code: str, date: date) -> pd.DataFrame:
        """
        获取指定日期之前的数据（避免未来函数）

        Args:
            code: 股票代码
            date: 截止日期

        Returns:
            历史数据DataFrame
        """
        return self.con.execute(f"""
            SELECT * FROM {self.TABLE_PRICES}
            WHERE code = ? AND date <= ?
            ORDER BY date
        """, [str(code), date]).df()

    def get_daily_data(self, date: date) -> pd.DataFrame:
        """
        获取指定日期的所有股票数据

        Args:
            date: 交易日期

        Returns:
            当日所有股票数据
        """
        return self.con.execute(f"""
            SELECT * FROM {self.TABLE_PRICES}
            WHERE date = ?
        """, [date]).df()
        
    def get_range_data(self, codes: Optional[List[str]] = None,
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> pd.DataFrame:
        """
        获取指定范围的批量数据(高性能方法)

        Args:
            codes: 股票代码列表(可选,默认全部)
            start_date: 开始日期(可选)
            end_date: 结束日期(可选)

        Returns:
            数据DataFrame
        """
        query = f"SELECT * FROM {self.TABLE_PRICES} WHERE 1=1"
        params = []

        if codes:
            placeholders = ','.join(['?' for _ in codes])
            query += f" AND code IN ({placeholders})"
            params.extend([str(c) for c in codes])

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY code, date"

        return self.con.execute(query, params).df()

    def get_all_data(self) -> pd.DataFrame:
        """
        获取所有日期的所有股票数据

        注意: 可能会返回大量数据，建议使用get_data_range()进行范围查询

        Returns:
            所有股票数据
        """
        query = f"SELECT * FROM {self.TABLE_PRICES}"
        df = self.con.execute(query).df()
        df.set_index(['code', 'date'], inplace=True)
        df.sort_index(inplace=True)

        return df

    def get_all_codes(self) -> List[str]:
        """
        获取所有股票代码

        Returns:
            股票代码列表（字符串）
        """
        # 确保所有代码都是字符串类型
        return [str(code) for code in self.stock_codes]

    def get_available_dates(self, start_date: Optional[date] = None,
                            end_date: Optional[date] = None) -> List[date]:
        """
        获取可用交易日期列表

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日期列表
        """
        dates = self.available_dates

        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        return dates

    def get_previous_trading_date(self, current_date: date, n: int = 1) -> Optional[date]:
        """
        获取指定日期前n个交易日的日期

        Args:
            current_date: 当前日期
            n: 向前推的交易日数量（默认为1，即前一个交易日）

        Returns:
            前n个交易日的日期，如果找不到则返回None

        Example:
            >>> # 获取2024-01-10前一个交易日
            >>> prev_date = data_handler.get_previous_trading_date(date(2024, 1, 10))
            >>> # 获取2024-01-10前3个交易日
            >>> prev_3_date = data_handler.get_previous_trading_date(date(2024, 1, 10), n=3)
        """
        # 获取所有交易日期
        all_dates = self.available_dates

        # 找到current_date在交易日期中的位置
        try:
            current_idx = all_dates.index(current_date)
        except ValueError:
            # current_date不是交易日，找到不晚于current_date的最后一个交易日
            valid_dates = [d for d in all_dates if d <= current_date]
            if not valid_dates:
                return None
            current_idx = all_dates.index(valid_dates[-1])

        # 计算目标索引
        target_idx = current_idx - n

        # 检查索引是否有效
        if target_idx < 0:
            return None

        return all_dates[target_idx]

    def update_data(self, new_data: pd.DataFrame) -> None:
        """
        更新数据（用于实盘）

        Args:
            new_data: 新的日线数据

        Note:
            自动过滤掉数据库表中不存在的列，避免列数不匹配错误
        """
        # 确保日期格式
        new_data['date'] = pd.to_datetime(new_data['date'])

        # 获取数据库表的列名
        table_columns = self.con.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{self.TABLE_PRICES}'
            ORDER BY ordinal_position
        """).fetchdf()['column_name'].tolist()

        # 只选择数据库表中存在的列
        available_columns = [col for col in table_columns if col in new_data.columns]

        if len(available_columns) < len(table_columns):
            missing_columns = set(table_columns) - set(available_columns)
            raise ValueError(f"数据缺少必需的列: {missing_columns}")

        # 选择需要的列并按表顺序排列
        filtered_data = new_data[available_columns].copy()

        # 确保 code 列是字符串类型，避免 DuckDB 类型推断问题
        if 'code' in filtered_data.columns:
            filtered_data['code'] = filtered_data['code'].astype('string')

        # 插入数据库
        self.con.register('temp_new_data', filtered_data)
        self.con.execute(f"INSERT OR REPLACE INTO {self.TABLE_PRICES} SELECT * FROM temp_new_data")
        self.con.unregister('temp_new_data')

        # 刷新元数据
        self._load_metadata()

    def load_index_data(self, index_path: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        加载指数数据（用于benchmark）

        Args:
            index_path: 指数数据文件路径（CSV格式，英文表头）
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）

        Returns:
            指数数据DataFrame，包含date和close列
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"指数数据文件不存在: {index_path}")

        # 读取指数数据
        index_df = pd.read_csv(index_path, encoding='utf-8')

        # 检查必要的列是否存在
        if 'date' not in index_df.columns or 'close' not in index_df.columns:
            raise ValueError(f"指数数据文件必须包含date和close列")

        # 转换日期格式
        index_df['date'] = pd.to_datetime(index_df['date'])

        # 过滤日期范围
        if start_date:
            index_df = index_df[index_df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            index_df = index_df[index_df['date'] <= pd.to_datetime(end_date)]

        # 只保留需要的列
        index_df = index_df[['date', 'close']].copy()

        # 按日期排序
        index_df = index_df.sort_values('date').reset_index(drop=True)

        return index_df

    def get_data_info(self) -> Dict:
        """
        获取数据集信息

        Returns:
            数据集信息字典
        """
        # 从数据库查询统计信息
        result = self.con.execute(f"""
            SELECT
                COUNT(DISTINCT code) as stock_count,
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(*) as total_records
            FROM {self.TABLE_PRICES}
        """).fetchdf()

        row = result.iloc[0]

        return {
            "status": "已连接",
            "stock_count": int(row['stock_count']),
            "start_date": str(row['min_date'].date()),
            "end_date": str(row['max_date'].date()),
            "trading_days": len(self.available_dates),
            "total_data_points": int(row['total_records'])
        }

    def __repr__(self) -> str:
        """字符串表示"""
        info = self.get_data_info()
        if info["status"] == "未连接":
            return "DataHandler(未连接)"

        return (f"DataHandler(股票: {info['stock_count']}, "
                f"日期: {info['start_date']} 至 {info['end_date']}, "
                f"交易日: {info['trading_days']})")

    def close(self) -> None:
        """关闭数据库连接"""
        if hasattr(self, 'con') and self.con:
            self.con.close()

    def execute_sql(self, sql: str, params: Optional[List] = None) -> pd.DataFrame:
        """
        执行自定义SQL查询

        Args:
            sql: SQL查询语句
            params: 查询参数(可选)

        Returns:
            查询结果DataFrame

        Example:
            >>> # 查询2024年涨幅最大的10只股票
            >>> sql = '''
            ... SELECT code,
            ...        (LAST(close) - FIRST(close)) / FIRST(close) as return_rate
            ... FROM stock_prices
            ... WHERE date >= '2024-01-01' AND date <= '2024-12-31'
            ... GROUP BY code
            ... ORDER BY return_rate DESC
            ... LIMIT 10
            ... '''
            >>> result = handler.execute_sql(sql)
        """
        if params:
            return self.con.execute(sql, params).df()
        else:
            return self.con.execute(sql).df()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    # ==================== 因子相关方法 ====================

    def get_stock_info(self, stock_code: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票基本信息

        Args:
            stock_code: 股票代码，None表示获取所有股票信息

        Returns:
            股票信息DataFrame
        """
        # 检查表是否存在
        table_exists = self.con.execute(f"""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = '{self.TABLE_STOCK_INFO}'
        """).fetchone()

        if not table_exists:
            return pd.DataFrame()

        if stock_code:
            return self.con.execute(f"""
                SELECT * FROM {self.TABLE_STOCK_INFO}
                WHERE code = ?
            """, [str(stock_code)]).df()
        else:
            return self.con.execute(f"""
                SELECT * FROM {self.TABLE_STOCK_INFO}
            """).df()

    def get_trading_days(self, start_date: Optional[date] = None,
                        end_date: Optional[date] = None) -> List[date]:
        """
        获取交易日历

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日期列表
        """
        # 检查表是否存在
        table_exists = self.con.execute(f"""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = '{self.TABLE_TRADING_DAYS}'
        """).fetchone()

        if not table_exists:
            # 如果交易日历表不存在，从价格表中获取
            return self.get_available_dates(start_date, end_date)

        query = f"SELECT date FROM {self.TABLE_TRADING_DAYS} WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        result = self.con.execute(query, params).fetchdf()
        return result['date'].tolist()

    def is_trading_day(self, date: date) -> bool:
        """
        检查指定日期是否为交易日

        Args:
            date: 日期

        Returns:
            是否为交易日
        """
        trading_days = self.get_trading_days(date, date)
        return date in trading_days

    def _init_factor_tables(self) -> None:
        """初始化因子表（如果不存在）"""
        # 创建序列用于生成因子 ID
        try:
            self.con.execute("CREATE SEQUENCE factor_id_seq START 1")
        except Exception:
            # 序列可能已存在，忽略错误
            pass

        # 创建因子定义表
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS factor_definitions (
                factor_id INTEGER PRIMARY KEY,
                factor_name VARCHAR(50) UNIQUE NOT NULL,
                factor_category VARCHAR(20),
                factor_desc TEXT,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建因子值表
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS factor_data (
                trade_date DATE NOT NULL,
                stock_code VARCHAR(10) NOT NULL,
                factor_id INTEGER NOT NULL,
                factor_value DOUBLE,
                PRIMARY KEY (trade_date, stock_code, factor_id),
                FOREIGN KEY (factor_id) REFERENCES factor_definitions(factor_id)
            )
        """)

        # 创建索引
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_factor_date_id ON factor_data(trade_date, factor_id)")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_factor_code_date ON factor_data(stock_code, trade_date)")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_factor_code_id_date ON factor_data(stock_code, factor_id, trade_date)")

    def register_factor(self, factor_id: int, factor_name: str,
                       factor_category: str = 'custom', factor_desc: str = None,
                       on_conflict: str = 'error') -> int:
        """
        注册因子定义

        Args:
            factor_id: 因子ID（必需）
            factor_name: 因子名称
            factor_category: 因子类别 (alpha158, custom, etc.)
            factor_desc: 因子描述
            on_conflict: 冲突处理策略
                - 'error': 抛出异常（默认）
                - 'skip': 跳过，返回已存在的ID
                - 'replace': 更新现有定义

        Returns:
            因子ID

        Raises:
            ValueError: 当 on_conflict='error' 且因子ID或名称已存在时
        """
        # 检查因子ID是否已存在
        existing_by_id = self.con.execute(
            f"SELECT factor_name FROM {self.TABLE_FACTOR_DEFINITIONS} WHERE factor_id = ?",
            [factor_id]
        ).fetchone()

        # 检查因子名称是否已存在
        existing_by_name = self.con.execute(
            f"SELECT factor_id FROM {self.TABLE_FACTOR_DEFINITIONS} WHERE factor_name = ?",
            [factor_name]
        ).fetchone()

        # 处理冲突
        if existing_by_id or existing_by_name:
            if on_conflict == 'error':
                if existing_by_id and existing_by_name:
                    if existing_by_id[0] == factor_name:
                        # 同一个因子，直接返回ID
                        return factor_id
                    else:
                        raise ValueError(
                            f"因子ID {factor_id} 已被因子 '{existing_by_id[0]}' 使用，"
                            f"且因子名 '{factor_name}' 已分配ID {existing_by_name[0]}"
                        )
                elif existing_by_id:
                    raise ValueError(
                        f"因子ID {factor_id} 已被因子 '{existing_by_id[0]}' 使用"
                    )
                else:
                    raise ValueError(
                        f"因子名 '{factor_name}' 已存在，ID为 {existing_by_name[0]}"
                    )
            elif on_conflict == 'skip':
                # 返回已存在的ID
                if existing_by_id:
                    return factor_id
                else:
                    return existing_by_name[0]
            elif on_conflict == 'replace':
                # 删除旧记录
                if existing_by_id:
                    self.con.execute(
                        f"DELETE FROM {self.TABLE_FACTOR_DEFINITIONS} WHERE factor_id = ?",
                        [factor_id]
                    )
                if existing_by_name and existing_by_name[0] != factor_id:
                    self.con.execute(
                        f"DELETE FROM {self.TABLE_FACTOR_DEFINITIONS} WHERE factor_name = ?",
                        [factor_name]
                    )

        # 插入新因子
        self.con.execute(f"""
            INSERT INTO {self.TABLE_FACTOR_DEFINITIONS} (factor_id, factor_name, factor_category, factor_desc)
            VALUES (?, ?, ?, ?)
        """, [factor_id, factor_name, factor_category, factor_desc])

        return factor_id

    def register_factor_auto(self, factor_name: str,
                            factor_category: str = 'custom',
                            factor_desc: str = None) -> int:
        """
        注册因子定义（自动生成ID，用于向后兼容和自动注册）

        Args:
            factor_name: 因子名称
            factor_category: 因子类别 (alpha158, custom, etc.)
            factor_desc: 因子描述

        Returns:
            因子ID
        """
        # 检查因子是否已存在
        result = self.con.execute(
            f"SELECT factor_id FROM {self.TABLE_FACTOR_DEFINITIONS} WHERE factor_name = ?",
            [factor_name]
        ).fetchone()

        if result:
            return result[0]

        # 插入新因子（使用序列生成 ID）
        self.con.execute(f"""
            INSERT INTO {self.TABLE_FACTOR_DEFINITIONS} (factor_id, factor_name, factor_category, factor_desc)
            VALUES (nextval('factor_id_seq'), ?, ?, ?)
        """, [factor_name, factor_category, factor_desc])

        # 返回新插入的ID
        result = self.con.execute(
            f"SELECT factor_id FROM {self.TABLE_FACTOR_DEFINITIONS} WHERE factor_name = ?",
            [factor_name]
        ).fetchone()
        return result[0] if result else None

    def get_factor_id(self, factor_name: str) -> Optional[int]:
        """
        获取因子ID

        Args:
            factor_name: 因子名称

        Returns:
            因子ID，如果不存在返回None
        """
        result = self.con.execute(
            f"SELECT factor_id FROM {self.TABLE_FACTOR_DEFINITIONS} WHERE factor_name = ?",
            [factor_name]
        ).fetchone()
        return result[0] if result else None

    def get_available_factors(self) -> List[str]:
        """
        获取所有可用的因子名称

        Returns:
            因子名称列表
        """
        result = self.con.execute(f"""
            SELECT factor_name FROM {self.TABLE_FACTOR_DEFINITIONS}
            ORDER BY factor_category, factor_name
        """).fetchdf()
        return result['factor_name'].tolist()

    def save_factors(self, factor_df: pd.DataFrame, trade_date: date) -> None:
        """
        批量保存因子值到数据库

        Args:
            factor_df: 因子DataFrame，索引为stock_code，列为因子名称
                      或者包含 'code' 列作为股票代码
            trade_date: 交易日期

        Example:
            >>> factors = pd.DataFrame({
            ...     'MA5': [1.02, 0.98, 1.01],
            ...     'MA10': [0.99, 1.02, 0.97]
            ... }, index=['000001', '000002', '600000'])
            >>> handler.save_factors(factors, date(2024, 1, 10))
        """
        # 如果 code 在列中而不是索引中，将其设置为索引
        if 'code' in factor_df.columns and factor_df.index.name != 'code':
            factor_df = factor_df.set_index('code')
        # 批量获取所有因子的ID（一次性查询，避免循环查询）
        factor_names = list(factor_df.columns)
        factor_ids = {}

        # 批量查询所有因子的ID
        placeholders = ','.join(['?' for _ in factor_names])
        query = f"SELECT factor_name, factor_id FROM {self.TABLE_FACTOR_DEFINITIONS} WHERE factor_name IN ({placeholders})"
        result = self.con.execute(query, factor_names).fetchdf()

        # 构建因子名到ID的映射
        for _, row in result.iterrows():
            factor_ids[row['factor_name']] = row['factor_id']

        # 准备数据：转换为长格式
        records = []
        for stock_code in factor_df.index:
            for factor_name in factor_df.columns:
                factor_value = factor_df.loc[stock_code, factor_name]
                if pd.notna(factor_value):  # 跳过NaN值
                    # 从缓存中获取因子ID
                    if factor_name not in factor_ids:
                        # 如果因子不存在，自动注册它（使用序列生成ID）
                        factor_ids[factor_name] = self.register_factor_auto(factor_name, 'custom')

                    factor_id = factor_ids[factor_name]

                    records.append({
                        'trade_date': trade_date,
                        'stock_code': str(stock_code),
                        'factor_id': factor_id,
                        'factor_value': float(factor_value)
                    })

        if not records:
            return

        # 批量插入
        insert_df = pd.DataFrame(records)
        # 显式将 stock_code 列设置为字符串类型，避免 DuckDB 类型推断为整数
        insert_df = insert_df.astype({'stock_code': 'string'})
        self.con.register('temp_factor_data', insert_df)
        self.con.execute(f"""
            INSERT OR REPLACE INTO {self.TABLE_FACTOR_DATA}
            SELECT
                trade_date,
                stock_code,
                factor_id,
                factor_value
            FROM temp_factor_data
        """)
        self.con.unregister('temp_factor_data')

    def get_factor_cross_section(self, trade_date: date,
                                factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取截面因子数据（某日所有股票的指定因子）

        Args:
            trade_date: 交易日期
            factor_names: 因子名称列表，None表示所有因子

        Returns:
            DataFrame，索引为stock_code，列为factor_names
        """
        # 构建查询
        if factor_names:
            factor_ids = []
            for name in factor_names:
                factor_id = self.get_factor_id(name)
                if factor_id:
                    factor_ids.append(factor_id)

            if not factor_ids:
                return pd.DataFrame()

            placeholders = ','.join(['?' for _ in factor_ids])
            query = f"""
                SELECT fd.stock_code, fd.factor_value, fdef.factor_name
                FROM {self.TABLE_FACTOR_DATA} fd
                JOIN {self.TABLE_FACTOR_DEFINITIONS} fdef ON fd.factor_id = fdef.factor_id
                WHERE fd.trade_date = ? AND fd.factor_id IN ({placeholders})
            """
            params = [trade_date] + factor_ids
        else:
            query = f"""
                SELECT fd.stock_code, fd.factor_value, fdef.factor_name
                FROM {self.TABLE_FACTOR_DATA} fd
                JOIN {self.TABLE_FACTOR_DEFINITIONS} fdef ON fd.factor_id = fdef.factor_id
                WHERE fd.trade_date = ?
            """
            params = [trade_date]

        result = self.con.execute(query, params).fetchdf()

        if result.empty:
            return pd.DataFrame()

        # 转换为宽表格式
        pivot_df = result.pivot(index='stock_code', columns='factor_name', values='factor_value')

        if factor_names:
            # 确保列顺序
            pivot_df = pivot_df.reindex(columns=factor_names)

        return pivot_df

    def get_stock_factors(self, stock_code: str,
                         factor_names: Optional[List[str]] = None,
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None) -> pd.DataFrame:
        """
        获取单只股票的时序因子数据

        Args:
            stock_code: 股票代码
            factor_names: 因子名称列表，None表示所有因子
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame，索引为trade_date，列为factor_names
        """
        # 确保 stock_code 是字符串类型
        stock_code = str(stock_code)

        # 构建查询
        if factor_names is not None:
            # 获取因子ID列表
            factor_ids = []
            for name in factor_names:
                factor_id = self.get_factor_id(name)
                if factor_id:
                    factor_ids.append(factor_id)

            if not factor_ids:
                return pd.DataFrame()

            placeholders = ','.join(['?' for _ in factor_ids])
            query = f"""
                SELECT fd.trade_date, fd.factor_value, fdef.factor_name
                FROM {self.TABLE_FACTOR_DATA} fd
                JOIN {self.TABLE_FACTOR_DEFINITIONS} fdef ON fd.factor_id = fdef.factor_id
                WHERE fd.stock_code = ? AND fd.factor_id IN ({placeholders})
            """
            params = [stock_code] + factor_ids
        else:
            query = f"""
                SELECT fd.trade_date, fd.factor_value, fdef.factor_name
                FROM {self.TABLE_FACTOR_DATA} fd
                JOIN {self.TABLE_FACTOR_DEFINITIONS} fdef ON fd.factor_id = fdef.factor_id
                WHERE fd.stock_code = ?
            """
            params = [stock_code]

        # 添加日期过滤
        if start_date:
            query += " AND fd.trade_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND fd.trade_date <= ?"
            params.append(end_date)

        query += " ORDER BY fd.trade_date"

        result = self.con.execute(query, params).fetchdf()

        if result.empty:
            return pd.DataFrame()

        # 转换为宽表格式
        pivot_df = result.pivot(index='trade_date', columns='factor_name', values='factor_value')

        if factor_names:
            # 确保列顺序
            pivot_df = pivot_df.reindex(columns=factor_names)

        return pivot_df

    def get_factors_wide(self, trade_date: date,
                        stock_codes: Optional[List[str]] = None,
                        factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取宽表格式的因子数据（用于ML预测）

        Args:
            trade_date: 交易日期
            stock_codes: 股票代码列表，None表示所有股票
            factor_names: 因子名称列表，None表示所有因子

        Returns:
            DataFrame，索引为stock_code，列为factor_names
        """
        # 如果没有指定因子，获取所有因子
        if factor_names is None:
            factor_names = self.get_available_factors()

        if not factor_names:
            return pd.DataFrame()

        # 构建PIVOT查询
        pivot_columns = ', '.join([
            f"MAX(CASE WHEN fdef.factor_name = '{name}' THEN fd.factor_value END) AS {name}"
            for name in factor_names
        ])

        query = f"""
            SELECT
                fd.stock_code,
                {pivot_columns}
            FROM {self.TABLE_FACTOR_DATA} fd
            JOIN {self.TABLE_FACTOR_DEFINITIONS} fdef ON fd.factor_id = fdef.factor_id
            WHERE fd.trade_date = ?
        """

        params = [trade_date]

        if stock_codes:
            placeholders = ','.join(['?' for _ in stock_codes])
            query += f" AND fd.stock_code IN ({placeholders})"
            params.extend([str(code) for code in stock_codes])

        query += " GROUP BY fd.stock_code"

        result = self.con.execute(query, params).fetchdf()

        if result.empty:
            return pd.DataFrame()

        result.set_index('stock_code', inplace=True)

        # 确保列顺序
        result = result.reindex(columns=factor_names)

        return result

    def get_factor_info(self) -> pd.DataFrame:
        """
        获取因子定义信息

        Returns:
            因子定义DataFrame
        """
        result = self.con.execute(f"""
            SELECT factor_id, factor_name, factor_category, factor_desc, created_time
            FROM {self.TABLE_FACTOR_DEFINITIONS}
            ORDER BY factor_category, factor_name
        """).fetchdf()

        return result

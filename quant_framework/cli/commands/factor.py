"""
因子计算命令模块

提供因子计算、分析等功能。
对应 Alpha158 和 MultiFactorAnalysis 类。
"""

import argparse
import pandas as pd
from tqdm import tqdm
from quant_framework.cli.commands.base import BaseCommand
from quant_framework.utils.logger import get_logger

logger = get_logger(__name__)


class FactorCommand(BaseCommand):
    """因子计算命令组"""

    @property
    def name(self) -> str:
        return "factor"

    @property
    def help(self) -> str:
        return "因子计算命令：计算、分析股票因子"

    def _add_subcommands(self):
        """添加因子相关的子命令"""
        subparsers = self.parser.add_subparsers(
            dest='action',
            title='可用子命令',
            description='因子计算相关的子命令'
        )

        # calculate 子命令
        calculate_parser = subparsers.add_parser(
            'calculate',
            help='计算因子',
            description='计算指定股票的因子值'
        )
        calculate_parser.add_argument(
            '--symbols',
            type=str,
            help='股票代码列表，用逗号分隔'
        )
        calculate_parser.add_argument(
            '--factor-name',
            type=str,
            default='alpha158',
            help='因子名称（默认：alpha158）'
        )
        calculate_parser.add_argument(
            '--date',
            type=str,
            help='日期（格式：YYYY-MM-DD）'
        )
        calculate_parser.add_argument(
            "--db-path",
            type=str,
            required=True,
            help='数据库路径',
        )
        calculate_parser.set_defaults(func=self.calculate)

        # analyze 子命令
        analyze_parser = subparsers.add_parser(
            'analyze',
            help='分析因子',
            description='分析因子的统计特性和预测能力'
        )
        analyze_parser.add_argument(
            '--factor-name',
            type=str,
            required=True,
            help='要分析的因子名称'
        )
        analyze_parser.add_argument(
            '--output',
            type=str,
            help='输出文件路径（可选）'
        )
        analyze_parser.set_defaults(func=self.analyze)

        # list 子命令
        list_parser = subparsers.add_parser(
            'list',
            help='列出可用因子',
            description='列出所有可用的因子类型'
        )
        list_parser.set_defaults(func=self.list_factors)

    def calculate(self, args: argparse.Namespace):
        logger.info("开始计算因子...")
        logger.info(f"数据库路径: {args.db_path}")
        logger.info(f"因子名称: {args.factor_name}")

        from quant_framework import DataHandler, DataUpdater, Alpha158
        from datetime import datetime
        from quant_framework.cli.utils import load_update_failures

        data_handler = DataHandler(args.db_path)
        data_updater = DataUpdater(data_handler)
        factor_calculator = Alpha158()

        # 获取股票代码列表
        if args.symbols:
            stock_code_list = args.symbols.split(',')
            logger.info(f"股票代码: {args.symbols}")
        else:
            stock_code_list = data_handler.get_all_codes()
            logger.info(f"计算全部股票的因子，共 {len(stock_code_list)} 只")

        # 获取目标日期
        if not args.date:
            target_date = data_updater.get_appropriate_end_date()
            logger.info(f"未设置计算日期，默认计算最新交易日的因子")
        else:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

        logger.info(f"计算日期: {target_date}")

        # 加载更新失败的股票列表
        update_failure_stocks = load_update_failures()
        if update_failure_stocks:
            logger.info(f"加载到 {len(update_failure_stocks)} 只更新失败的股票，将跳过")

        # 注册因子定义
        for factor_id, factor_name, factor_category, factor_desc in Alpha158.DEFINITIONS:
            data_handler.register_factor(factor_id, factor_name, factor_category, factor_desc)
        logger.info(f"已注册 {len(Alpha158.DEFINITIONS)} 个因子定义")

        # 计算因子日期范围（前推60个交易日）
        factor_end_date = target_date
        factor_start_date = data_handler.get_previous_trading_date(factor_end_date, 60)
        logger.info(f"数据范围: {factor_start_date} 到 {factor_end_date}")

        # 计算因子
        factor_df = None
        skipped_count = 0
        calculated_count = 0

        logger.info("开始计算因子")
        pbar = tqdm(stock_code_list)
        for code in pbar:
            pbar.set_postfix({"code": code, "skip": skipped_count, "calc": calculated_count})

            # 跳过更新失败的股票
            if code in update_failure_stocks:
                skipped_count += 1
                continue

            # 检查是否已存在该交易日的因子记录
            existing_factors = data_handler.get_stock_factors(
                stock_code=code,
                start_date=factor_end_date,
                end_date=factor_end_date
            )

            if not existing_factors.empty:
                skipped_count += 1
                continue
            
            # 检查交易数据是否更新
            earliest_date, latest_date = data_handler.get_stock_latest_date(code)
            
            if latest_date < target_date:
                skipped_count += 1
                continue    

            # 获取股票历史数据
            df = data_handler.get_range_data([code], factor_start_date, factor_end_date)
            
            # 计算新因子（只取最后一行）
            try:
                new_factor_row = factor_calculator.calculate(df).iloc[[-1]]
            except Exception as e:
                logger.warning(f"股票 {code} 因子计算失败: {e}")
                skipped_count += 1
                continue

            if factor_df is None:
                factor_df = new_factor_row
            else:
                factor_df = pd.concat([factor_df, new_factor_row], ignore_index=False)

            calculated_count += 1

        if factor_df is not None:
            factor_df = factor_df.set_index('code')

        logger.info(f"因子计算完成: 共 {len(stock_code_list)} 只股票，"
                    f"跳过 {skipped_count} 只，新计算 {calculated_count} 只")

        # 保存因子
        if factor_df is not None:
            factor_names = [d[1] for d in Alpha158.DEFINITIONS]
            data_handler.save_factors(factor_df[factor_names], factor_end_date)
            logger.info(f"因子已保存到数据库")
        else:
            logger.warning("没有需要保存的因子数据")

    def analyze(self, args: argparse.Namespace):
        """
        分析因子

        Args:
            args: 命令行参数
        """
        logger.info(f"分析因子: {args.factor_name}")
        if args.output:
            logger.info(f"输出文件: {args.output}")

        # TODO: 实现具体的因子分析逻辑
        # 示例：
        # from quant_framework import MultiFactorAnalysis
        # analyzer = MultiFactorAnalysis()
        # results = analyzer.analyze(args.factor_name)
        # if args.output:
        #     analyzer.save_results(results, args.output)

        logger.info("因子分析功能待实现")

    def list_factors(self, args: argparse.Namespace):
        """
        列出可用因子

        Args:
            args: 命令行参数
        """
        logger.info("可用的因子类型：")
        # TODO: 实现具体的因子列表查询逻辑
        # 示例：
        # from quant_framework import FactorRegistry
        # registry = FactorRegistry()
        # factors = registry.list_all()
        # for factor in factors:
        #     print(f"  - {factor['name']}: {factor['description']}")

        logger.info("  - alpha158: Alpha158 因子")
        logger.info("因子列表查询功能待完善")

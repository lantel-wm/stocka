"""
数据管理命令模块

提供数据更新、初始化、状态查看等功能。
对应 DataHandler 和 DataUpdater 类。
"""

import os
import argparse
from quant_framework.cli.commands.base import BaseCommand
from quant_framework.utils.logger import get_logger

logger = get_logger(__name__)

class DataCommand(BaseCommand):
    """数据管理命令组"""

    @property
    def name(self) -> str:
        return "data"

    @property
    def help(self) -> str:
        return "数据管理命令：更新、初始化、查看数据状态"

    def _add_subcommands(self):
        """添加数据相关的子命令"""
        subparsers = self.parser.add_subparsers(
            dest='action',
            title='可用子命令',
            description='数据管理相关的子命令'
        )

        # update 子命令
        update_parser = subparsers.add_parser(
            'update',
            help='更新股票数据',
            description='从数据源更新股票数据到数据库'
        )
        update_parser.add_argument(
            '--symbols',
            type=str,
            help='股票代码列表，用逗号分隔（如 000001,600000），默认更新全部股票'
        )
        update_parser.add_argument(
            '--date',
            type=str,
            help='更新日期（格式：YYYY-MM-DD）'
        )

        update_parser.add_argument(
            "--db-path",
            type=str,
            required=True,
            help='数据库路径',
        )
        
        update_parser.set_defaults(func=self.update)

        # status 子命令
        status_parser = subparsers.add_parser(
            'status',
            help='查看数据状态',
            description='查看数据库中的数据状态和统计信息'
        )
        status_parser.add_argument(
            '--detail',
            action='store_true',
            help='显示详细信息'
        )
        status_parser.set_defaults(func=self.status)

        # init 子命令
        init_parser = subparsers.add_parser(
            'init',
            help='初始化数据库',
            description='初始化数据库和必要的表结构'
        )
        init_parser.add_argument(
            '--force',
            action='store_true',
            help='强制重新初始化（会删除现有数据）'
        )
        init_parser.set_defaults(func=self.init)

    def update(self, args: argparse.Namespace):
        """
        更新股票数据

        Args:
            args: 命令行参数
        """
        logger.info("开始更新股票数据...")
        logger.info(f"数据库路径: {args.db_path}")
        
        from quant_framework import DataHandler, DataUpdater
        from datetime import date, datetime
        
        data_handler = DataHandler(args.db_path)
        data_updater = DataUpdater(data_handler)
        
        if args.symbols:
            stock_code_list = args.symbols
            logger.info(f"股票代码: {args.symbols}")
        else:
            stock_code_list = data_handler.get_all_codes()
            logger.info(f"更新全部股票")
        
        if not args.date:
            target_date = data_updater.get_appropriate_end_date()
            logger.info(f"未设置更新时间，默认更新最新交易日的数据")
        else:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
            
        logger.info(f"更新日期: {target_date}")
        
        
        update_result = data_updater.update_batch_stock_data(
            stock_codes=stock_code_list, end_date=target_date.strftime('%Y%m%d')
        )

        update_failure_stocks = update_result['failed_stocks']
        logger.info(f"更新失败列表: {update_failure_stocks}")

        # 保存失败股票列表，供后续因子计算使用
        from quant_framework.cli.utils import save_update_failures
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        save_update_failures(update_failure_stocks, target_date)

        logger.info("数据更新完成")


    def status(self, args: argparse.Namespace):
        """
        查看数据状态

        Args:
            args: 命令行参数
        """
        logger.info("查看数据状态...")
        if args.detail:
            logger.info("显示详细信息")

        # TODO: 实现具体的状态查询逻辑
        # 示例：
        # from quant_framework import DataHandler
        # handler = DataHandler()
        # stats = handler.get_statistics()
        # print(json.dumps(stats, indent=2))

        logger.info("数据状态查询功能待实现")

    def init(self, args: argparse.Namespace):
        """
        初始化数据库

        Args:
            args: 命令行参数
        """
        logger.info("初始化数据库...")
        if args.force:
            logger.warning("强制重新初始化：将删除现有数据")

        # TODO: 实现具体的初始化逻辑
        # 示例：
        # from quant_framework import DataHandler
        # handler = DataHandler()
        # handler.initialize(force=args.force)

        logger.info("数据库初始化功能待实现")

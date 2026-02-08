"""
CLI 命令基类

提供统一的命令接口，所有具体命令模块都应继承此类。
"""

import argparse
from abc import ABC, abstractmethod
from typing import Optional


class BaseCommand(ABC):
    """
    命令基类

    所有命令组（data, factor, ml, live 等）都应该继承此类并实现相应方法。
    """

    def __init__(self, subparsers: argparse._SubParsersAction):
        """
        初始化命令

        Args:
            subparsers: argparse 的子解析器对象，用于添加子命令
        """
        self.subparsers = subparsers
        self.parser = self._create_parser()
        self._add_subcommands()

    @property
    @abstractmethod
    def name(self) -> str:
        """
        命令名称

        Returns:
            命令名称（如 'data', 'factor', 'ml', 'live'）
        """
        pass

    @property
    @abstractmethod
    def help(self) -> str:
        """
        命令帮助信息

        Returns:
            帮助文本
        """
        pass

    def _create_parser(self) -> argparse.ArgumentParser:
        """
        创建命令解析器

        Returns:
            ArgumentParser 对象
        """
        parser = self.subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.help
        )
        return parser

    @abstractmethod
    def _add_subcommands(self):
        """
        添加子命令

        在此方法中为当前命令组添加所有子命令。
        例如，对于 data 命令组，可以添加 update, status 等子命令。

        实现示例:
            # 添加 update 子命令
            update_parser = self.parser.add_subparsers(dest='action')
            update_cmd = update_parser.add_parser('update', help='更新数据')
            update_cmd.set_defaults(func=self.update)

        Args:
            无
        """
        pass

    # 以下是占位符方法，子类可以根据需要实现具体逻辑

    def run(self, args: argparse.Namespace):
        """
        执行命令

        Args:
            args: 命令行参数
        """
        # 如果有 action 属性，调用对应的方法
        if hasattr(args, 'func') and args.func:
            args.func(args)
        else:
            self.parser.print_help()

    # 子类可以实现的具体命令方法示例
    # def update(self, args: argparse.Namespace):
    #     """更新数据"""
    #     raise NotImplementedError("请实现此方法")

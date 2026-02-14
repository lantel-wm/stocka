"""
CLI 主逻辑

解析命令行参数并分发到对应的命令处理模块。
"""

import argparse
import sys
from quant_framework.utils.logger import get_logger
from quant_framework.cli.commands.data import DataCommand
from quant_framework.cli.commands.factor import FactorCommand
from quant_framework.cli.commands.ml import MLCommand
from quant_framework.cli.commands.live import LiveCommand
from quant_framework.cli.commands.strategy import StrategyCommand

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器

    Returns:
        ArgumentParser 对象
    """
    parser = argparse.ArgumentParser(
        prog='python -m quant_framework.cli',
        description='Stocka 量化交易框架命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python -m quant_framework.cli data update --db-path data/stock.db
  python -m quant_framework.cli factor calculate --db-path data/stock.db
  python -m quant_framework.cli ml train --model-name lightgbm
  python -m quant_framework.cli live update --interval 60
  python -m quant_framework.cli strategy predict --db-path data/stock.db --model-path ckpt/model.pkl

更多信息请参考文档。
        """
    )

    # 添加全局选项
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细日志'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='Stocka Quant Framework v1.0.0'
    )

    # 创建子命令解析器
    subparsers = parser.add_subparsers(
        dest='command',
        title='可用命令',
        description='量化交易相关的命令组',
        help='选择要执行的命令组'
    )

    # 注册所有命令组
    commands = [
        DataCommand(subparsers),
        FactorCommand(subparsers),
        MLCommand(subparsers),
        LiveCommand(subparsers),
        StrategyCommand(subparsers),
    ]

    # 存储命令实例以便后续使用
    parser._commands = {cmd.name: cmd for cmd in commands}

    return parser


def main():
    """
    CLI 主入口函数

    解析命令行参数并执行相应的命令。
    """
    parser = create_parser()
    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logger.setLevel('DEBUG')
        logger.debug("详细日志模式已启用")

    # 如果没有指定命令，显示帮助信息
    if not args.command:
        parser.print_help()
        return 0

    # 执行对应的命令
    try:
        command = parser._commands.get(args.command)
        if command:
            command.run(args)
        else:
            logger.error(f"未知命令: {args.command}")
            return 1
    except KeyboardInterrupt:
        logger.info("\n操作已取消")
        return 130
    except Exception as e:
        logger.error(f"执行命令时出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
交易日调度器

处理交易日判断、定时任务执行等功能。
"""

import os
import sys
import logging
import subprocess
from datetime import datetime, date
from typing import Optional, Callable
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from quant_framework.utils.constraints import TradingCalendar
from quant_framework.utils.logger import get_logger
from quant_framework.utils.config import Config
from quant_framework.notification import create_notifier
from quant_framework.reporting import ReportGenerator

logger = get_logger(__name__)


class TradingScheduler:
    """交易日调度器

    在交易日执行routine任务，并将结果推送到指定渠道。

    Example:
        >>> scheduler = TradingScheduler(config_path='config.yaml')
        >>> scheduler.run_daily_task()
    """

    def __init__(self, config_path: str = 'notify_config.yaml',
                 routine_script: str = './routine.sh',
                 signals_dir: str = 'signals'):
        """
        Args:
            config_path: 配置文件路径（默认使用 notify_config.yaml）
            routine_script: routine.sh脚本路径
            signals_dir: 预测结果存放目录
        """
        self.config_path = config_path
        self.routine_script = routine_script
        self.signals_dir = signals_dir
        self.calendar = TradingCalendar()

        # 加载配置
        try:
            self.config = Config(config_path).config
        except Exception as e:
            logger.warning(f"无法加载配置文件 {config_path}: {e}")
            self.config = {}

    def is_trading_day(self, target_date: Optional[date] = None) -> bool:
        """判断是否为交易日

        Args:
            target_date: 目标日期，默认为今天

        Returns:
            bool: 是交易日返回True
        """
        if target_date is None:
            target_date = datetime.now().date()

        return self.calendar.is_trading_day(target_date)

    def run_routine(self) -> bool:
        """运行routine.sh脚本

        Returns:
            bool: 执行成功返回True
        """
        logger.info("开始执行routine.sh...")

        if not os.path.exists(self.routine_script):
            logger.error(f"routine.sh脚本不存在: {self.routine_script}")
            return False

        try:
            # 执行脚本
            result = subprocess.run(
                ['bash', self.routine_script],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=1800  # 30分钟超时
            )

            # 输出日志
            if result.stdout:
                logger.info("routine.sh输出:\n" + result.stdout)
            if result.stderr:
                logger.warning("routine.sh错误输出:\n" + result.stderr)

            if result.returncode == 0:
                logger.info("routine.sh执行成功")
                return True
            else:
                logger.error(f"routine.sh执行失败，返回码: {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("routine.sh执行超时（30分钟）")
            return False
        except Exception as e:
            logger.error(f"routine.sh执行异常: {e}")
            return False

    def send_notification(self) -> bool:
        """发送预测报告通知

        Returns:
            bool: 发送成功返回True
        """
        logger.info("开始发送通知...")

        # 查找最新报告
        generator = ReportGenerator()
        csv_path = generator.find_latest_report(self.signals_dir)

        if not csv_path:
            logger.error(f"未在 {self.signals_dir} 中找到预测报告")
            return False

        logger.info(f"找到报告: {csv_path}")

        # 加载报告
        try:
            report = generator.load_from_csv(csv_path)
        except Exception as e:
            logger.error(f"加载报告失败: {e}")
            return False

        # 生成HTML报告
        html_content = generator.to_html(report, style='detailed')

        # 获取默认渠道列表（兼容新旧配置格式）
        default_channels = self.config.get('default_channels', ['email'])

        success_count = 0

        for channel in default_channels:
            # 从根级别获取配置（新格式）
            notifier_config = self.config.get(channel, {})

            # 如果没找到，尝试从 notification 嵌套层获取（旧格式兼容）
            if not notifier_config:
                notification_config = self.config.get('notification', {})
                notifier_config = notification_config.get(channel, {})

            if not notifier_config or not notifier_config.get('enabled', False):
                logger.warning(f"{channel} 未启用，跳过")
                continue

            try:
                notifier = create_notifier(channel, notifier_config)
                success = notifier.send_html(
                    subject=f"【选股日报】{report.date_str} 推荐 {report.count} 只股票",
                    content=html_content,
                    attachments=[csv_path]
                )

                if success:
                    logger.info(f"✅ {channel} 通知发送成功")
                    success_count += 1
                else:
                    logger.error(f"❌ {channel} 通知发送失败")

            except Exception as e:
                logger.error(f"❌ {channel} 通知发送异常: {e}")

        return success_count > 0

    def run_daily_task(self, force: bool = False) -> bool:
        """执行每日任务

        完整的每日任务流程：
        1. 判断是否为交易日
        2. 执行routine.sh
        3. 发送通知

        Args:
            force: 强制运行（忽略交易日判断）

        Returns:
            bool: 任务整体成功返回True
        """
        today = datetime.now().date()
        logger.info(f"开始执行每日任务: {today}")

        # 判断是否为交易日
        if not force and not self.is_trading_day(today):
            logger.info(f"{today} 不是交易日，跳过任务")
            return True

        logger.info(f"{today} 是交易日，开始执行任务")

        # 执行routine
        if not self.run_routine():
            # routine失败，发送告警通知
            self._send_alert(f"routine.sh执行失败，日期: {today}")
            return False

        # 发送通知
        if not self.send_notification():
            logger.error("通知发送失败")
            return False

        logger.info("每日任务执行完成")
        return True

    def _send_alert(self, message: str):
        """发送告警通知

        Args:
            message: 告警消息
        """
        logger.error(f"告警: {message}")

        # 尝试发送告警邮件（使用备用配置）
        try:
            # 从根级别获取（新格式），如果没有则从 notification 嵌套层获取（旧格式）
            email_config = self.config.get('email', {})
            if not email_config:
                notification_config = self.config.get('notification', {})
                email_config = notification_config.get('email', {})

            if email_config.get('enabled'):
                notifier = create_notifier('email', email_config)
                notifier.send_text(
                    subject=f"【Stocka告警】{datetime.now().strftime('%m-%d %H:%M')}",
                    content=message
                )
        except Exception as e:
            logger.error(f"发送告警失败: {e}")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='Stocka 交易日调度器')
    parser.add_argument(
        '--config',
        default='notify_config.yaml',
        help='配置文件路径（默认: notify_config.yaml）'
    )
    parser.add_argument(
        '--routine-script',
        default='./routine.sh',
        help='routine.sh脚本路径'
    )
    parser.add_argument(
        '--signals-dir',
        default='signals',
        help='预测结果目录'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制运行（忽略交易日判断）'
    )
    parser.add_argument(
        '--check-trading-day',
        action='store_true',
        help='仅检查今天是否为交易日'
    )

    args = parser.parse_args()

    scheduler = TradingScheduler(
        config_path=args.config,
        routine_script=args.routine_script,
        signals_dir=args.signals_dir
    )

    if args.check_trading_day:
        is_trading = scheduler.is_trading_day()
        print(f"今天是交易日: {'是' if is_trading else '否'}")
        sys.exit(0 if is_trading else 1)

    success = scheduler.run_daily_task(force=args.force)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

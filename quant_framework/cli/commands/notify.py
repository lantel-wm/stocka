"""
通知命令模块

提供邮件、飞书等通知渠道的测试和发送功能。
"""

import os
import argparse
from datetime import datetime
from typing import Optional

from quant_framework.cli.commands.base import BaseCommand
from quant_framework.utils.logger import get_logger
from quant_framework.utils.config import Config
from quant_framework.notification import create_notifier, MessageFormat
from quant_framework.reporting import ReportGenerator

logger = get_logger(__name__)


class NotifyCommand(BaseCommand):
    """通知命令组"""

    @property
    def name(self) -> str:
        return "notify"

    @property
    def help(self) -> str:
        return "通知命令：测试、发送邮件/飞书/微信"

    def _add_subcommands(self):
        """添加通知相关的子命令"""
        subparsers = self.parser.add_subparsers(
            dest='action',
            title='可用子命令',
            description='通知相关的子命令'
        )

        # test 子命令 - 测试通知渠道
        test_parser = subparsers.add_parser(
            'test',
            help='测试通知渠道',
            description='发送测试消息验证配置是否正确'
        )
        test_parser.add_argument(
            '--channel',
            type=str,
            default='email',
            choices=['email', 'feishu', 'wechat', 'all'],
            help='通知渠道 (默认: email)'
        )
        test_parser.add_argument(
            '--config',
            type=str,
            default='notify_config.yaml',
            help='配置文件路径 (默认: notify_config.yaml)'
        )
        test_parser.set_defaults(func=self.test)

        # send 子命令 - 发送预测报告
        send_parser = subparsers.add_parser(
            'send',
            help='发送选股预测报告',
            description='将选股预测结果发送到指定渠道'
        )
        send_parser.add_argument(
            '--date',
            type=str,
            help='报告日期 (格式: YYYY-MM-DD 或 YYYYMMDD)，默认最新报告'
        )
        send_parser.add_argument(
            '--channel',
            type=str,
            default='email',
            choices=['email', 'feishu', 'wechat', 'all'],
            help='通知渠道 (默认: email)'
        )
        send_parser.add_argument(
            '--signals-dir',
            type=str,
            default='signals',
            help='预测结果目录 (默认: signals)'
        )
        send_parser.add_argument(
            '--config',
            type=str,
            default='notify_config.yaml',
            help='配置文件路径 (默认: notify_config.yaml)'
        )
        send_parser.add_argument(
            '--dry-run',
            action='store_true',
            help='仅生成报告不发送'
        )
        send_parser.set_defaults(func=self.send)

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            config = Config(config_path)
            return config.config
        except Exception as e:
            logger.warning(f"无法加载配置文件 {config_path}: {e}")
            return {}

    def _get_notifier_config(self, config: dict, channel: str) -> Optional[dict]:
        """获取指定渠道的配置"""
        # 先从根级别获取（新配置格式）
        channel_config = config.get(channel, {})
        if channel_config:
            return channel_config

        # 再从 notification 嵌套层获取（兼容旧配置格式）
        notification_config = config.get('notification', {})
        return notification_config.get(channel, {})

    def test(self, args: argparse.Namespace):
        """
        测试通知渠道

        Args:
            args: 命令行参数
        """
        logger.info(f"开始测试通知渠道: {args.channel}")

        # 加载配置
        config = self._load_config(args.config)

        if args.channel == 'all':
            # 从配置中获取启用的渠道列表
            channels = config.get('default_channels', ['email'])
        else:
            channels = [args.channel]

        for channel in channels:
            notifier_config = self._get_notifier_config(config, channel)
            if not notifier_config or not notifier_config.get('enabled', False):
                logger.warning(f"{channel} 未在配置中启用")
                continue

            try:
                from quant_framework.notification import create_notifier
                logger.info(f"正在创建 {channel} 通知器...")
                notifier = create_notifier(channel, notifier_config)
                logger.info(f"✓ 通知器创建成功，enabled={notifier.enabled}")

                # 测试直接调用底层SMTP发送
                if channel == 'email':
                    import smtplib
                    import ssl
                    from email.mime.text import MIMEText
                    from email.header import Header

                    email_config = notifier_config
                    smtp_server = email_config.get('smtp_server')
                    smtp_port = int(email_config.get('smtp_port', 465))
                    username = email_config.get('username')
                    password = email_config.get('password')
                    from_addr = email_config.get('from_addr')
                    to_addrs = email_config.get('to_addrs', [])

                    logger.info(f"测试直接SMTP连接: {smtp_server}:{smtp_port}")
                    logger.info(f"发件人: {username}, 收件人: {to_addrs}")

                    try:
                        from email.mime.multipart import MIMEMultipart
                        import uuid

                        context = ssl.create_default_context()
                        if email_config.get('use_ssl', True):
                            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30, context=context)
                        else:
                            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
                            server.starttls(context=context)

                        logger.info("SMTP连接建立，正在登录...")
                        server.login(username, password)
                        logger.info(f"✓ 登录成功: {username}")

                        # 构建规范的邮件
                        msg = MIMEMultipart('alternative')
                        msg['From'] = from_addr  # 不使用Header包装，避免编码问题
                        msg['To'] = ', '.join(to_addrs)
                        msg['Subject'] = f"Stocka SMTP测试 - {datetime.now().strftime('%m-%d %H:%M')}"
                        msg['Message-ID'] = f"<stocka-{uuid.uuid4()}@stocka.local>"
                        msg['Date'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0800')

                        # 添加邮件正文
                        text_content = f"""这是一封Stocka量化框架的测试邮件。

发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
发件人: {from_addr}
收件人: {', '.join(to_addrs)}

如果收到此邮件，说明SMTP配置正确！

---
Stocka 量化交易框架
"""
                        msg.attach(MIMEText(text_content, 'plain', 'utf-8'))

                        logger.info(f"正在发送邮件...")
                        result = server.sendmail(from_addr, to_addrs, msg.as_string())
                        if result:
                            logger.warning(f"部分收件人发送失败: {result}")
                        else:
                            logger.info(f"✓ 邮件服务器确认发送成功")
                        server.quit()
                        logger.info("✅ 直接SMTP测试发送成功")
                    except Exception as smtp_e:
                        logger.error(f"❌ 直接SMTP测试失败: {smtp_e}")
                        raise

            except Exception as e:
                logger.error(f"❌ {channel} 测试失败: {e}")
                import traceback
                logger.error(f"异常详情: {traceback.format_exc()}")

    def send(self, args: argparse.Namespace):
        """
        发送选股预测报告

        Args:
            args: 命令行参数
        """
        logger.info("开始发送选股预测报告...")

        # 加载配置
        config = self._load_config(args.config)

        # 创建报告生成器
        generator = ReportGenerator()

        # 确定要发送的报告文件
        if args.date:
            # 从日期构建文件名
            date_str = args.date.replace('-', '')
            # 查找匹配的CSV文件
            import glob
            pattern = os.path.join(args.signals_dir, f"{date_str}*.csv")
            matching_files = glob.glob(pattern)
            if not matching_files:
                logger.error(f"未找到日期 {args.date} 的预测报告")
                return
            csv_path = matching_files[0]
        else:
            # 查找最新报告
            csv_path = generator.find_latest_report(args.signals_dir)
            if not csv_path:
                logger.error(f"未在 {args.signals_dir} 目录中找到预测报告")
                return

        logger.info(f"使用报告文件: {csv_path}")

        # 加载报告
        try:
            report = generator.load_from_csv(csv_path)
        except Exception as e:
            logger.error(f"加载报告失败: {e}")
            return

        # 生成HTML报告
        html_content = generator.to_html(report, style='detailed')
        text_summary = generator.generate_summary(report)

        # 仅生成不发送
        if args.dry_run:
            logger.info("干运行模式，仅生成报告")
            print("\n" + "=" * 50)
            print(text_summary)
            print("=" * 50)
            return

        # 发送报告
        channels = config.get('default_channels', ['email']) if args.channel == 'all' else [args.channel]

        for channel in channels:
            notifier_config = self._get_notifier_config(config, channel)
            if not notifier_config or not notifier_config.get('enabled', False):
                logger.warning(f"{channel} 未在配置中启用，跳过")
                continue

            try:
                notifier = create_notifier(channel, notifier_config)

                # 发送HTML报告
                success = notifier.send_html(
                    subject=f"【选股日报】{report.date_str} 推荐 {report.count} 只股票",
                    content=html_content,
                    attachments=[csv_path]
                )

                if success:
                    logger.info(f"✅ 报告已通过 {channel} 发送成功")
                else:
                    logger.error(f"❌ 报告通过 {channel} 发送失败")

            except Exception as e:
                logger.error(f"❌ 通过 {channel} 发送失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())

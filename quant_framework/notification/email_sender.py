"""
邮件发送模块

支持SMTP协议发送邮件，包括HTML格式和附件。
"""

import os
import smtplib
import ssl
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.header import Header
from pathlib import Path
from typing import Dict, Any, List

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseNotifier, NotificationMessage, MessageFormat

logger = logging.getLogger(__name__)


class EmailNotifier(BaseNotifier):
    """SMTP邮件发送器

    支持SSL/TLS加密，支持HTML格式和附件。

    Config Example:
        {
            "enabled": true,
            "smtp_server": "smtp.qq.com",
            "smtp_port": 465,
            "use_ssl": true,
            "username": "your_email@qq.com",
            "password": "your_auth_code",  # 建议使用环境变量
            "from_addr": "your_email@qq.com",
            "to_addrs": ["recipient@example.com"],
            "subject_template": "【选股日报】{date} 推荐 {count} 只股票"
        }
    """

    # 必需的配置字段
    REQUIRED_FIELDS = ["smtp_server", "smtp_port", "username", "from_addr", "to_addrs"]

    def __init__(self, config: Dict[str, Any]):
        """初始化邮件发送器

        Args:
            config: 邮件配置字典
        """
        super().__init__(config)
        self.smtp_server = config.get("smtp_server")
        self.smtp_port = int(config.get("smtp_port", 465))
        self.use_ssl = config.get("use_ssl", True)
        self.username = config.get("username")
        # 优先从环境变量读取密码
        self.password = config.get("password") or os.getenv("EMAIL_PASSWORD", "")
        self.from_addr = config.get("from_addr")
        self.to_addrs = self._parse_to_addrs(config.get("to_addrs", []))
        self.subject_template = config.get("subject_template", "【选股日报】{date}")

    def _parse_to_addrs(self, to_addrs: Any) -> List[str]:
        """解析收件人地址列表"""
        if isinstance(to_addrs, str):
            return [addr.strip() for addr in to_addrs.split(",") if addr.strip()]
        elif isinstance(to_addrs, list):
            return [addr for addr in to_addrs if addr]
        return []

    def validate_config(self) -> bool:
        """验证邮件配置"""
        if not self.enabled:
            return True

        missing = []
        for field in self.REQUIRED_FIELDS:
            value = getattr(self, field, None)
            if not value:
                missing.append(field)

        if missing:
            raise ValueError(f"邮件配置缺少必需字段: {', '.join(missing)}")

        if not self.password:
            logger.warning("邮件密码未配置，将从环境变量 EMAIL_PASSWORD 读取")

        if not self.to_addrs:
            raise ValueError("收件人列表 (to_addrs) 不能为空")

        # 打印配置信息（隐藏密码）
        logger.info(f"邮件配置验证通过 -> 发件人: {self.from_addr}, 收件人: {self.to_addrs}")
        logger.info(f"SMTP服务器: {self.smtp_server}:{self.smtp_port}, SSL: {self.use_ssl}")

        return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def send(self, message: NotificationMessage) -> bool:
        """发送邮件

        Args:
            message: 要发送的消息对象

        Returns:
            bool: 发送成功返回True
        """
        if not self.enabled:
            logger.info("邮件通知器已禁用，跳过发送")
            return False

        logger.info(f"开始发送邮件 -> 主题: {message.subject}")
        self.validate_config()

        try:
            msg = self._build_message(message)
            self._send_via_smtp(msg)
            logger.info(f"✓ 邮件发送流程完成: {message.subject}")
            return True

        except Exception as e:
            logger.error(f"✗ 邮件发送失败: {e}")
            raise

    def _build_message(self, message: NotificationMessage) -> MIMEMultipart:
        """构建邮件消息"""
        import uuid
        from datetime import datetime

        msg = MIMEMultipart()

        # 对纯ASCII地址不使用Header包装，避免编码问题
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)
        msg["Subject"] = message.subject

        # 添加必要的邮件头
        msg["Message-ID"] = f"<stocka-{uuid.uuid4()}@stocka.local>"
        msg["Date"] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0800')
        msg["MIME-Version"] = "1.0"

        # 根据格式选择内容类型
        if message.format == MessageFormat.HTML:
            content_type = "html"
            subtype = "html"
        elif message.format == MessageFormat.MARKDOWN:
            # Markdown转为纯文本发送，或可以扩展为转为HTML
            content_type = "plain"
            subtype = "plain"
        else:
            content_type = "plain"
            subtype = "plain"

        # 添加邮件正文
        body = MIMEText(message.content, subtype, "utf-8")
        msg.attach(body)

        # 添加附件
        if message.attachments:
            for attachment_path in message.attachments:
                self._attach_file(msg, attachment_path)

        return msg

    def _attach_file(self, msg: MIMEMultipart, filepath: str):
        """添加附件到邮件"""
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"附件不存在: {filepath}")
            return

        try:
            with open(path, "rb") as f:
                attachment = MIMEApplication(f.read())
                attachment.add_header(
                    "Content-Disposition",
                    f"attachment; filename=\"{path.name}\""
                )
                msg.attach(attachment)
                logger.debug(f"附件已添加: {path.name}")
        except Exception as e:
            logger.error(f"添加附件失败 {filepath}: {e}")

    def _send_via_smtp(self, msg: MIMEMultipart):
        """通过SMTP发送邮件"""
        context = ssl.create_default_context()

        # 打印收件人信息
        logger.info(f"准备发送邮件 -> 收件人: {self.to_addrs}")

        if self.use_ssl:
            # SSL连接（端口465）
            logger.debug(f"使用SSL连接 {self.smtp_server}:{self.smtp_port}")
            server = smtplib.SMTP_SSL(
                host=self.smtp_server,
                port=self.smtp_port,
                timeout=30,
                context=context
            )
        else:
            # TLS连接（端口587）
            logger.debug(f"使用TLS连接 {self.smtp_server}:{self.smtp_port}")
            server = smtplib.SMTP(
                host=self.smtp_server,
                port=self.smtp_port,
                timeout=30
            )
            server.starttls(context=context)

        try:
            # 登录
            logger.info(f"正在登录发件邮箱: {self.username}")
            server.login(self.username, self.password)
            logger.info(f"✓ 邮箱登录成功: {self.username}")

            # 发送邮件
            server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            logger.info(f"✓ 邮件已发送至: {', '.join(self.to_addrs)}")
        finally:
            server.quit()
            logger.debug("SMTP连接已关闭")

    def send_with_template(self, date_str: str, count: int, content: str,
                          format: MessageFormat = MessageFormat.HTML,
                          attachments: List[str] = None) -> bool:
        """使用模板发送邮件

        Args:
            date_str: 日期字符串
            count: 股票数量
            content: 邮件正文内容
            format: 内容格式
            attachments: 附件列表
        """
        subject = self.subject_template.format(date=date_str, count=count)
        message = NotificationMessage(
            subject=subject,
            content=content,
            format=format,
            attachments=attachments or []
        )
        return self.send(message)

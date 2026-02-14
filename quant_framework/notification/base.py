"""
通知模块基类

提供统一的通知接口，支持多种推送渠道。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class MessageFormat(Enum):
    """消息格式类型"""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class NotificationMessage:
    """通知消息数据类"""
    subject: str
    content: str
    format: MessageFormat = MessageFormat.TEXT
    attachments: Optional[List[str]] = None  # 附件路径列表
    extra: Optional[Dict[str, Any]] = None  # 额外信息


class BaseNotifier(ABC):
    """通知器抽象基类

    所有具体通知器（邮件、飞书、微信等）都需要继承此类。

    Example:
        >>> class EmailNotifier(BaseNotifier):
        ...     def send(self, message: NotificationMessage) -> bool:
        ...         # 实现邮件发送逻辑
        ...         pass
        ...
        >>> notifier = EmailNotifier(config)
        >>> message = NotificationMessage(
        ...     subject="测试",
        ...     content="这是一封测试邮件",
        ...     format=MessageFormat.HTML
        ... )
        >>> notifier.send(message)
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化通知器

        Args:
            config: 配置字典，包含发送所需的参数
        """
        self.config = config
        self.enabled = config.get("enabled", True)

    @abstractmethod
    def send(self, message: NotificationMessage) -> bool:
        """发送通知

        Args:
            message: 要发送的消息对象

        Returns:
            bool: 发送成功返回True，失败返回False
        """
        pass

    def send_text(self, subject: str, content: str, **kwargs) -> bool:
        """发送纯文本消息（便捷方法）"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"send_text 被调用: subject={subject[:30]}...")

        message = NotificationMessage(
            subject=subject,
            content=content,
            format=MessageFormat.TEXT,
            **kwargs
        )
        logger.info(f"调用 send 方法...")
        result = self.send(message)
        logger.info(f"send 方法返回: {result}")
        return result

    def send_markdown(self, subject: str, content: str, **kwargs) -> bool:
        """发送Markdown格式消息（便捷方法）"""
        message = NotificationMessage(
            subject=subject,
            content=content,
            format=MessageFormat.MARKDOWN,
            **kwargs
        )
        return self.send(message)

    def send_html(self, subject: str, content: str, **kwargs) -> bool:
        """发送HTML格式消息（便捷方法）"""
        message = NotificationMessage(
            subject=subject,
            content=content,
            format=MessageFormat.HTML,
            **kwargs
        )
        return self.send(message)

    def is_enabled(self) -> bool:
        """检查通知器是否启用"""
        return self.enabled

    def validate_config(self) -> bool:
        """验证配置是否有效

        Returns:
            bool: 配置有效返回True

        Raises:
            ValueError: 配置无效时抛出异常，说明缺少哪些必需字段
        """
        return True

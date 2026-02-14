"""
通知模块

支持多种推送渠道：邮件、飞书、企业微信等。

Usage:
    from quant_framework.notification import create_notifier, NotificationMessage

    # 创建邮件发送器
    notifier = create_notifier('email', config)

    # 发送消息
    message = NotificationMessage(
        subject='测试',
        content='<h1>Hello</h1>',
        format=MessageFormat.HTML
    )
    notifier.send(message)
"""

from .base import BaseNotifier, NotificationMessage, MessageFormat
from .email_sender import EmailNotifier

# 可用的通知器映射
NOTIFIER_REGISTRY = {
    'email': EmailNotifier,
    # 后续可添加：
    # 'feishu': FeishuNotifier,
    # 'wechat': WechatNotifier,
    # 'dingtalk': DingtalkNotifier,
}


def create_notifier(channel: str, config: dict) -> BaseNotifier:
    """创建通知器实例

    Args:
        channel: 通知渠道名称 ('email', 'feishu', etc.)
        config: 配置字典

    Returns:
        BaseNotifier: 通知器实例

    Raises:
        ValueError: 如果渠道不支持
    """
    if channel not in NOTIFIER_REGISTRY:
        raise ValueError(f"不支持的通知渠道: {channel}. "
                        f"可用渠道: {list(NOTIFIER_REGISTRY.keys())}")

    notifier_class = NOTIFIER_REGISTRY[channel]
    return notifier_class(config)


def get_available_channels() -> list:
    """获取所有可用的通知渠道"""
    return list(NOTIFIER_REGISTRY.keys())


__all__ = [
    'BaseNotifier',
    'NotificationMessage',
    'MessageFormat',
    'EmailNotifier',
    'create_notifier',
    'get_available_channels',
]

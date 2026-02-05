"""
日志配置模块
提供统一的日志配置和获取logger的函数
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# 默认配置
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_LOG_FILE = 'logs/quant_framework.log'
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5

# 环境变量名称
ENV_LOG_FILE = 'STOCKA_LOG_FILE'
ENV_LOG_LEVEL = 'STOCKA_LOG_LEVEL'


# 存储已配置的loggers
_configured_loggers = set()


def setup_logger(
    name: str,
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    console: bool = True,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT
) -> logging.Logger:
    """
    配置并返回一个logger实例

    Args:
        name: logger名称，通常使用__name__
        level: 日志级别，默认为INFO。也可通过环境变量STOCKA_LOG_LEVEL设置
        log_file: 日志文件路径，None则使用环境变量STOCKA_LOG_FILE（如果设置）
        console: 是否输出到控制台，默认True
        max_bytes: 日志文件最大大小，默认10MB
        backup_count: 保留的备份文件数量，默认5

    Returns:
        配置好的logger实例

    Examples:
        >>> from quant_framework.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")

    使用环境变量:
        >>> # 在shell中设置环境变量
        >>> export STOCKA_LOG_FILE='logs/app.log'
        >>> export STOCKA_LOG_LEVEL='DEBUG'
        >>> # 所有模块的日志都会输出到logs/app.log
    """
    logger = logging.getLogger(name)

    # 如果已经配置过，直接返回
    if name in _configured_loggers:
        return logger

    # 优先使用环境变量的配置
    if log_file is None and ENV_LOG_FILE in os.environ:
        log_file = os.environ[ENV_LOG_FILE]

    if level is None and ENV_LOG_LEVEL in os.environ:
        level_str = os.environ[ENV_LOG_LEVEL].upper()
        level = getattr(logging, level_str, DEFAULT_LOG_LEVEL)

    # 设置日志级别
    log_level = level or DEFAULT_LOG_LEVEL
    logger.setLevel(log_level)

    # 清除已有的handlers
    logger.handlers.clear()

    # 创建formatter
    formatter = logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT
    )

    # 添加控制台handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 添加文件handler
    if log_file:
        log_path = Path(log_file)
        # 创建日志目录
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 防止日志传播到父logger
    logger.propagate = False

    # 标记为已配置
    _configured_loggers.add(name)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    获取logger实例的便捷函数

    Args:
        name: logger名称，如果为None则使用调用者的模块名

    Returns:
        logger实例

    Examples:
        >>> from quant_framework.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("信息日志")
        >>> logger.warning("警告日志")
        >>> logger.error("错误日志")
    """
    if name is None:
        # 获取调用者的模块名
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'root')

    return setup_logger(name)


# 预配置一些常用的logger
def get_logger_for_module(module_name: str) -> logging.Logger:
    """
    为特定模块获取logger

    Args:
        module_name: 模块名称

    Returns:
        logger实例
    """
    return get_logger(module_name)

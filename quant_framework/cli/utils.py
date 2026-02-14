"""
CLI 辅助工具模块

提供命令行工具的通用辅助功能。
"""

import json
import os
from datetime import date
from typing import List, Optional
from quant_framework.utils.logger import get_logger

logger = get_logger(__name__)

# 默认的配置目录
DEFAULT_CONFIG_DIR = os.path.expanduser("~/.stocka")
# 失败股票列表文件
UPDATE_FAILURES_FILE = "update_failures.json"


def get_failures_file_path() -> str:
    """
    获取失败股票列表文件的完整路径

    Returns:
        文件完整路径
    """
    if not os.path.exists(DEFAULT_CONFIG_DIR):
        os.makedirs(DEFAULT_CONFIG_DIR)
    return os.path.join(DEFAULT_CONFIG_DIR, UPDATE_FAILURES_FILE)


def save_update_failures(failed_stocks: List[str], target_date: date) -> None:
    """
    保存更新失败的股票列表

    Args:
        failed_stocks: 失败的股票代码列表
        target_date: 更新目标日期
    """
    file_path = get_failures_file_path()
    data = {
        "date": target_date.strftime("%Y-%m-%d"),
        "failed_stocks": failed_stocks
    }

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.debug(f"已保存更新失败列表到 {file_path}")


def load_update_failures() -> List[str]:
    """
    读取最近一次更新失败的股票列表

    Returns:
        失败的股票代码列表，如果没有记录则返回空列表
    """
    file_path = get_failures_file_path()

    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("failed_stocks", [])
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"读取失败列表文件出错: {e}")
        return []


def clear_update_failures() -> None:
    """
    清除失败股票列表文件
    """
    file_path = get_failures_file_path()
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.debug(f"已清除失败列表文件 {file_path}")

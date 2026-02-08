"""
CLI 命令模块

此包包含所有的 CLI 命令实现。
每个命令模块对应一个功能领域（data, factor, ml, live 等）。
"""

from .base import BaseCommand

__all__ = ['BaseCommand']

"""
配置管理模块
负责读取和管理回测配置文件
"""

import yaml
import os
import importlib
from typing import Dict, Any, Optional, Type
from ..strategy.base_strategy import BaseStrategy


class Config:
    """
    配置类
    管理回测框架的所有配置参数
    """

    # 策略模块路径
    STRATEGY_MODULE_PATH = "quant_framework.strategy"

    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def reload(self) -> None:
        """重新加载配置文件"""
        self.config = self._load_config()

    def save(self, save_path: Optional[str] = None) -> str:
        """
        保存配置到文件

        Args:
            save_path: 保存路径（默认为原配置文件路径）

        Returns:
            保存的文件路径
        """
        path = save_path or self.config_path

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)

        return path

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的嵌套键）

        Args:
            key: 配置键（支持 'data.base_path' 格式）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        设置配置值（支持点号分隔的嵌套键）

        Args:
            key: 配置键（支持 'data.base_path' 格式）
            value: 配置值
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """返回配置的字典副本"""
        return self.config.copy()

    # ========== 便捷方法 ==========

    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.get('data', {})

    def get_backtest_config(self) -> Dict[str, Any]:
        """获取回测配置"""
        return self.get('backtest', {})

    def get_strategy_config(self) -> Dict[str, Any]:
        """获取策略配置"""
        return self.get('strategy', {})

    def get_transaction_cost_config(self) -> Dict[str, Any]:
        """获取交易成本配置"""
        return self.get('transaction_cost', {})

    def get_risk_control_config(self) -> Dict[str, Any]:
        """获取风险控制配置"""
        return self.get('risk_control', {})

    def get_performance_config(self) -> Dict[str, Any]:
        """获取绩效分析配置"""
        return self.get('performance', {})

    def get_output_config(self) -> Dict[str, Any]:
        """获取输出配置"""
        return self.get('output', {})

    # ========== 策略创建方法 ==========

    def create_strategy(self) -> BaseStrategy:
        """
        根据配置创建策略实例（通过类名动态导入）

        Returns:
            策略实例

        Raises:
            ValueError: 如果策略类不存在或导入失败
        """
        strategy_config = self.get_strategy_config()
        strategy_type = strategy_config.get('type', 'SimpleMAStrategy')
        strategy_params = strategy_config.get('params', {})

        try:
            # 动态导入策略模块
            strategy_module = importlib.import_module(self.STRATEGY_MODULE_PATH)

            # 根据类名获取策略类
            if hasattr(strategy_module, strategy_type):
                strategy_class = getattr(strategy_module, strategy_type)
            else:
                raise ValueError(f"策略类 '{strategy_type}' 在模块 {self.STRATEGY_MODULE_PATH} 中不存在")

            # 创建策略实例
            strategy = strategy_class(strategy_params)

            return strategy

        except ImportError as e:
            raise ValueError(f"导入策略模块失败: {e}")
        except Exception as e:
            raise ValueError(f"创建策略实例失败 (strategy_type={strategy_type}): {e}")

    def __repr__(self) -> str:
        """字符串表示"""
        return f"Config(path={self.config_path})"


def load_config(config_path: str = "config.yaml") -> Config:
    """
    加载配置文件的便捷函数

    Args:
        config_path: 配置文件路径

    Returns:
        Config实例
    """
    return Config(config_path)

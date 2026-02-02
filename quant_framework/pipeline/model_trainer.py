"""
模型训练器
负责数据准备、模型训练和保存
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
import pandas as pd

from ..data.data_handler import DataHandler
from ..model.lgb_model import LGBModel
from .data_loader import DataLoader


class ModelTrainer:
    """
    模型训练器

    负责完整的模型训练流程：
    1. 使用外部传入的DataLoader加载训练集和验证集
    2. 训练模型
    3. 保存模型
    """

    def __init__(self,
                 factors: list,
                 model_params: Optional[Dict] = None):
        """
        初始化模型训练器

        Args:
            factors: 因子列名列表
            model_params: 模型参数（可选）
        """
        self.factors = factors

        # 默认模型参数
        default_params = {
            'loss': 'mse',
            'num_boost_round': 1000,
            'early_stopping_rounds': 50,
        }

        self.model_params = default_params
        if model_params:
            self.model_params.update(model_params)

        self.model = None
        self.train_df = None
        self.valid_df = None

    def prepare_data(self, train_loader, valid_loader) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        准备训练和验证数据

        Args:
            train_loader: 训练集 DataLoader
            valid_loader: 验证集 DataLoader

        Returns:
            (训练数据, 验证数据)
        """
        print("\n准备训练数据...")

        # 加载训练集（每个DataLoader独立进行截面标准化）
        self.train_df = train_loader.load()

        # 加载验证集（每个DataLoader独立进行截面标准化）
        self.valid_df = valid_loader.load()

        return self.train_df, self.valid_df

    def train(self) -> LGBModel:
        """
        训练模型

        Returns:
            训练好的LGBModel实例
        """
        if self.train_df is None or self.valid_df is None:
            self.prepare_data()

        print("\n开始训练模型...")

        # 创建模型
        self.model = LGBModel(
            factors=self.factors,
            loss=self.model_params['loss'],
            num_boost_round=self.model_params['num_boost_round'],
            early_stopping_rounds=self.model_params['early_stopping_rounds']
        )

        # 训练模型
        evals_result = self.model.fit(
            train_df=self.train_df,
            valid_df=self.valid_df,
            verbose_eval=20
        )

        # 获取最佳迭代轮数
        if 'valid' in evals_result and 'l2' in evals_result['valid']:
            best_iteration = len(evals_result['valid']['l2'])
            print(f"\n✓ 模型训练完成，迭代轮数: {best_iteration}")
        else:
            print(f"\n✓ 模型训练完成")

        return self.model

    def save_model(self, output_dir: str, model_name: str = "model") -> Path:
        """
        保存模型到文件

        Args:
            output_dir: 输出目录
            model_name: 模型文件名（不含扩展名）

        Returns:
            模型文件路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_path = output_path / model_name
        self.model.save_model(str(model_path), save_format='pkl')

        return model_path.with_suffix('.pkl')

    def get_training_info(self) -> Dict:
        """
        获取训练信息

        Returns:
            训练信息字典
        """
        if self.model is None:
            return {
                'status': '未训练',
                'train_samples': len(self.train_df) if self.train_df is not None else 0,
                'valid_samples': len(self.valid_df) if self.valid_df is not None else 0,
            }

        info = {
            'status': '已训练',
            'train_samples': len(self.train_df),
            'valid_samples': len(self.valid_df),
            'factors': self.factors,
            'model_params': self.model_params,
        }

        # 添加模型信息
        model_info = self.model.get_model_info()
        info.update(model_info)

        return info

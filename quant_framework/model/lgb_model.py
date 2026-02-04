"""
LightGBM 模型定义
用于量化金融的回归和分类任务
"""

import pickle
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class LGBModel:
    """
    LightGBM 模型包装类
    支持训练、预测、保存和加载
    """

    def __init__(self, factors: list, loss="mse", early_stopping_rounds=50, num_boost_round=1000, **kwargs):
        """
        初始化 LightGBM 模型

        Args:
            factors: 因子列名列表
            loss: 损失函数类型 ("mse" 或 "binary")
            early_stopping_rounds: 早停轮数
            num_boost_round: 最大迭代轮数
            **kwargs: 传递给 lightgbm 的其他参数
        """
        if loss not in {"mse", "binary"}:
            raise NotImplementedError(f"Unsupported loss type: {loss}")

        self.factors = factors
        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None
        self.training_info = {}  # 存储训练信息

    def _prepare_data(self, df: pd.DataFrame, reweighter=None) -> lgb.Dataset:
        """
        准备训练数据

        Args:
            df: 包含因子和标签的DataFrame
            reweighter: 样本权重（可选）

        Returns:
            lgb.Dataset
        """
        x, y = df[self.factors], df["LABEL0"]

        # LightGBM 需要 1D 数组作为标签
        if y.values.ndim == 1:
            pass
        elif y.values.ndim == 2 and y.values.shape[1] == 1:
            y = np.squeeze(y.values)
        else:
            raise ValueError("LightGBM doesn't support multi-label training")

        if reweighter is None:
            w = None
        else:
            raise ValueError("Unsupported reweighter type.")

        return lgb.Dataset(x.values, label=y, weight=w, free_raw_data=False)

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        num_boost_round: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose_eval: int = 20,
        evals_result: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        """
        训练模型

        Args:
            train_df: 训练数据
            valid_df: 验证数据
            num_boost_round: 最大迭代轮数（可选，覆盖初始化参数）
            early_stopping_rounds: 早停轮数（可选，覆盖初始化参数）
            verbose_eval: 日志输出频率
            evals_result: 存储评估结果的字典
            **kwargs: 传递给 lgb.train 的其他参数

        Returns:
            评估结果字典
        """
        if evals_result is None:
            evals_result = {}

        train_dataset = self._prepare_data(train_df)
        valid_dataset = self._prepare_data(valid_df)

        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds if early_stopping_rounds is None else early_stopping_rounds
        )
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(evals_result)

        self.model = lgb.train(
            self.params,
            train_dataset,
            num_boost_round=self.num_boost_round if num_boost_round is None else num_boost_round,
            valid_sets=[train_dataset, valid_dataset],
            valid_names=['train', 'valid'],
            callbacks=[early_stopping_callback, verbose_eval_callback, evals_result_callback],
            **kwargs,
        )

        # 保存训练信息
        self.training_info = {
            'evals_result': evals_result,
            'params': self.params,
            'num_boost_round': num_boost_round if num_boost_round is not None else self.num_boost_round,
            'early_stopping_rounds': early_stopping_rounds if early_stopping_rounds is not None else self.early_stopping_rounds,
            'factors': self.factors,
        }

        return evals_result

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """
        预测

        Args:
            test_df: 测试数据

        Returns:
            预测结果 Series
        """
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        x_test = test_df[self.factors]
        predictions = self.model.predict(x_test.values)

        return pd.Series(predictions, index=x_test.index)

    def save_model(self, model_path: str, save_format: str = "pkl"):
        """
        保存模型到文件

        Args:
            model_path: 模型保存路径（不带扩展名）
            save_format: 保存格式
                - "pkl": pickle格式（推荐，包含完整对象）
                - "txt": lightgbm原生格式（仅模型参数）
                - "both": 同时保存两种格式
        """
        if self.model is None:
            raise ValueError("Cannot save unfitted model!")

        model_path = Path(model_path)

        if save_format in ["pkl", "both"]:
            # 使用 pickle 保存完整对象（包括训练信息）
            pkl_path = model_path.with_suffix('.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"模型已保存到: {pkl_path}")

        if save_format in ["txt", "both"]:
            # 使用 lightgbm 原生格式（只保存模型，不保存训练信息）
            txt_path = model_path.with_suffix('.txt')
            self.model.save_model(str(txt_path))
            logger.info(f"模型已保存到: {txt_path}")

    @classmethod
    def load_model(cls, model_path: str) -> 'LGBModel':
        """
        从文件加载模型

        Args:
            model_path: 模型文件路径（.pkl 或 .txt）

        Returns:
            LGBModel 实例
        """
        model_path = Path(model_path)

        if model_path.suffix == '.pkl':
            # 从 pickle 加载完整对象
            with open(model_path, 'rb') as f:
                model_instance = pickle.load(f)

            logger.info(f"模型已从 {model_path} 加载")
            if hasattr(model_instance, 'model') and model_instance.model is not None:
                logger.info(f"  - 训练轮数: {model_instance.model.num_trees()}")

            return model_instance

        elif model_path.suffix == '.txt':
            # 从 lightgbm 原生格式加载
            booster = lgb.Booster(model_file=str(model_path))
            model_instance = cls()  # 创建新实例（使用默认参数）
            model_instance.model = booster

            logger.info(f"模型已从 {model_path} 加载")
            logger.info(f"  - 训练轮数: {booster.num_trees()}")

            return model_instance

        else:
            raise ValueError(f"Unsupported model file format: {model_path}. Use .pkl or .txt")

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            包含模型信息的字典
        """
        if self.model is None:
            return {"status": "未训练"}

        info = {
            "status": "已训练",
            "num_trees": self.model.num_trees(),
            "params": self.params,
        }

        if self.training_info:
            info['training_info'] = self.training_info

        return info

    def __repr__(self) -> str:
        """字符串表示"""
        factor_count = len(self.factors) if self.factors else 0
        if self.model is None:
            return f"LGBModel(未训练, factors={factor_count}, objective={self.params['objective']})"

        return (f"LGBModel(已训练, trees={self.model.num_trees()}, "
                f"factors={factor_count}, objective={self.params['objective']})")

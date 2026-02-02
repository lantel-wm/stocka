"""
模型评估器
负责模型测试和因子有效性指标计算（IC、RankIC等）
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
from scipy.stats import spearmanr

from ..data.data_handler import DataHandler
from ..model.lgb_model import LGBModel


def str_to_date(date_str: str) -> date:
    """将字符串转换为日期对象"""
    return datetime.strptime(date_str, '%Y-%m-%d').date()


class ModelEvaluator:
    """
    模型评估器

    计算因子有效性指标：
    - IC (Information Coefficient): 预测值与真实值的相关系数
    - RankIC: 预测值与真实值的Spearman相关系数
    - IC均度: IC值的标准差
    """

    def __init__(self, model: LGBModel):
        """
        初始化模型评估器

        Args:
            model: 训练好的模型
        """
        self.model = model
        self.predictions = None
        self.metrics = {}

    def predict(self, test_loader) -> pd.DataFrame:
        """
        在测试集上进行预测

        Args:
            test_loader: 测试集 DataLoader

        Returns:
            包含预测值和真实值的DataFrame
        """
        print("\n开始模型预测...")

        # 使用 test_loader 加载数据（已经计算好标签）
        test_data = test_loader.load()

        # 提取预测特征和真实标签
        X_test = test_data[self.model.factors]
        y_test = test_data['LABEL0']
        dates = test_data['date'] if 'date' in test_data.columns else None

        # 预测
        pred_values = self.model.predict(X_test)

        # 构建结果 DataFrame
        all_predictions = []
        for idx, code in enumerate(pred_values.index):
            all_predictions.append({
                'date': dates.iloc[idx] if dates is not None else None,
                'code': code,
                'prediction': pred_values.iloc[idx],
                'actual': y_test.iloc[idx],
            })

        # 转换为DataFrame
        self.predictions = pd.DataFrame(all_predictions)

        print(f"  - 预测样本数: {len(self.predictions)}")

        return self.predictions

    def calculate_ic(self, by_date: bool = True) -> Dict[str, float]:
        """
        计算IC指标

        Args:
            by_date: 是否按日期计算IC（默认True，截面IC）

        Returns:
            IC指标字典
        """
        if self.predictions is None:
            self.predict()

        print("\n计算IC指标...")

        if by_date:
            # 截面IC：每天计算一次IC
            daily_ic = []

            for date in self.predictions['date'].unique():
                date_data = self.predictions[self.predictions['date'] == date]

                if len(date_data) < 2:  # 至少需要2个样本
                    continue

                # Pearson相关系数
                ic = date_data['prediction'].corr(date_data['actual'])
                daily_ic.append(ic)

            daily_ic = pd.Series(daily_ic)

            # 计算IC统计量
            ic_mean = daily_ic.mean()
            ic_std = daily_ic.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0  # IC信息比率

            self.metrics.update({
                'IC_mean': ic_mean,
                'IC_std': ic_std,
                'IC_IR': ic_ir,
            })

            print(f"  - IC均值: {ic_mean:.4f}")
            print(f"  - IC标准差: {ic_std:.4f}")
            print(f"  - IC_IR: {ic_ir:.4f}")

        else:
            # 时序IC：整体计算
            ic = self.predictions['prediction'].corr(self.predictions['actual'])
            self.metrics['IC'] = ic
            print(f"  - IC: {ic:.4f}")

        return self.metrics

    def calculate_rank_ic(self) -> Dict[str, float]:
        """
        计算RankIC（Spearman相关系数）

        Returns:
            RankIC指标字典
        """
        if self.predictions is None:
            self.predict()

        print("\n计算RankIC指标...")

        # 按日期计算RankIC
        daily_rank_ic = []

        for date in self.predictions['date'].unique():
            date_data = self.predictions[self.predictions['date'] == date]

            if len(date_data) < 2:
                continue

            # Spearman相关系数
            rank_ic, _ = spearmanr(date_data['prediction'], date_data['actual'])
            if not np.isnan(rank_ic):
                daily_rank_ic.append(rank_ic)

        daily_rank_ic = pd.Series(daily_rank_ic)

        # 计算RankIC统计量
        rank_ic_mean = daily_rank_ic.mean()
        rank_ic_std = daily_rank_ic.std()
        rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0

        self.metrics.update({
            'RankIC_mean': rank_ic_mean,
            'RankIC_std': rank_ic_std,
            'RankIC_IR': rank_ic_ir,
        })

        print(f"  - RankIC均值: {rank_ic_mean:.4f}")
        print(f"  - RankIC标准差: {rank_ic_std:.4f}")
        print(f"  - RankIC_IR: {rank_ic_ir:.4f}")

        return self.metrics

    def calculate_metrics(self) -> Dict[str, float]:
        """
        计算所有评估指标

        Returns:
            所有指标的字典
        """
        if self.predictions is None:
            self.predict()

        self.calculate_ic(by_date=True)
        self.calculate_rank_ic()

        return self.metrics

    def save_evaluation_report(self, output_dir: str) -> Path:
        """
        保存评估报告到JSON

        Args:
            output_dir: 输出目录

        Returns:
            报告文件路径
        """
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_path = output_path / "evaluation_report.json"

        # 保存指标
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 评估报告已保存到: {report_path}")

        return report_path

    def get_evaluation_summary(self) -> str:
        """
        获取评估摘要文本

        Returns:
            评估摘要字符串
        """
        if not self.metrics:
            self.calculate_metrics()

        summary = []
        summary.append("=" * 60)
        summary.append("模型评估报告")
        summary.append("=" * 60)

        summary.append("\nIC指标:")
        summary.append(f"  IC均值:       {self.metrics.get('IC_mean', 0):.4f}")
        summary.append(f"  IC标准差:     {self.metrics.get('IC_std', 0):.4f}")
        summary.append(f"  IC_IR:        {self.metrics.get('IC_IR', 0):.4f}")

        summary.append("\nRankIC指标:")
        summary.append(f"  RankIC均值:   {self.metrics.get('RankIC_mean', 0):.4f}")
        summary.append(f"  RankIC标准差: {self.metrics.get('RankIC_std', 0):.4f}")
        summary.append(f"  RankIC_IR:    {self.metrics.get('RankIC_IR', 0):.4f}")

        summary.append("=" * 60)

        return "\n".join(summary)

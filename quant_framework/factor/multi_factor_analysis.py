"""
多因子分析模块
提供多因子组合、因子正交化、因子合成等功能
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .factor_metrics import FactorMetrics


class MultiFactorAnalysis:
    """
    多因子分析类

    提供多因子组合、因子正交化、因子合成等功能
    """

    def __init__(self, factors: Dict[str, pd.DataFrame]):
        """
        初始化多因子分析

        Args:
            factors: 因子字典，格式为 {因子名: 因子DataFrame}
                     因子DataFrame的索引为日期，列为股票代码
        """
        self.factors = factors
        self.factor_names = list(factors.keys())

    def combine_factors(self,
                       weights: Optional[Dict[str, float]] = None,
                       method: str = 'weighted_sum') -> pd.DataFrame:
        """
        合并多个因子

        Args:
            weights: 各因子权重（字典，格式为 {因子名: 权重}）
            method: 合并方法
                - 'weighted_sum': 加权求和
                - 'equal_weight': 等权平均
                - 'ic_weighted': IC加权（使用IC作为权重）
                - 'rank_weighted': 排名加权

        Returns:
            合并后的因子DataFrame
        """
        # 获取共同的日期和股票
        common_dates = None
        common_stocks = None

        for factor_name, factor_df in self.factors.items():
            if common_dates is None:
                common_dates = factor_df.index
            else:
                common_dates = common_dates.intersection(factor_df.index)

            if common_stocks is None:
                common_stocks = factor_df.columns
            else:
                common_stocks = common_stocks.intersection(factor_df.columns)

        if len(common_dates) == 0 or len(common_stocks) == 0:
            raise ValueError("因子之间没有共同的日期或股票")

        # 对齐所有因子
        aligned_factors = {}
        for factor_name, factor_df in self.factors.items():
            aligned_factors[factor_name] = factor_df.loc[common_dates, common_stocks]

        # 确定权重
        if method == 'equal_weight':
            weights = {name: 1.0 / len(self.factors) for name in self.factor_names}

        elif method == 'weighted_sum':
            if weights is None:
                raise ValueError("weights参数不能为空（当method='weighted_sum'时）")
            # 标准化权重
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

        elif method == 'ic_weighted':
            # 使用IC的绝对值作为权重
            # 这里简化处理，实际应该传入历史IC数据
            weights = {name: 1.0 for name in self.factor_names}
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

        elif method == 'rank_weighted':
            weights = {name: 1.0 / len(self.factors) for name in self.factor_names}

        else:
            raise ValueError(f"Unknown method: {method}")

        # 合并因子
        combined_factor = pd.DataFrame(0.0, index=common_dates, columns=common_stocks)

        for factor_name, weight in weights.items():
            if method == 'rank_weighted':
                # 排名加权：先将因子转换为排名，再加权求和
                factor_rank = aligned_factors[factor_name].rank(axis=1, pct=True)
                combined_factor += factor_rank * weight
            else:
                combined_factor += aligned_factors[factor_name] * weight

        return combined_factor

    def orthogonalize_factors(self,
                             reference_factor: str,
                             method: str = 'regression') -> Dict[str, pd.DataFrame]:
        """
        因子正交化（剔除参考因子的影响）

        Args:
            reference_factor: 参考因子名称
            method: 正交化方法
                - 'regression': 回归法（取残差）
                - 'gram_schmidt': Gram-Schmidt正交化
                - 'pca': PCA主成分分析

        Returns:
            正交化后的因子字典
        """
        if reference_factor not in self.factor_names:
            raise ValueError(f"参考因子 {reference_factor} 不存在")

        if method == 'regression':
            return self._orthogonalize_regression(reference_factor)
        elif method == 'gram_schmidt':
            return self._orthogonalize_gram_schmidt(reference_factor)
        elif method == 'pca':
            return self._orthogonalize_pca(reference_factor)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _orthogonalize_regression(self, reference_factor: str) -> Dict[str, pd.DataFrame]:
        """
        使用回归法进行因子正交化

        对每个因子，将其对参考因子做回归，取残差作为正交化后的因子
        """
        orthogonalized = {}

        # 获取参考因子
        ref_factor = self.factors[reference_factor]

        for factor_name, factor_df in self.factors.items():
            if factor_name == reference_factor:
                # 参考因子不变
                orthogonalized[factor_name] = factor_df.copy()
                continue

            # 对齐数据
            common_dates = factor_df.index.intersection(ref_factor.index)
            common_stocks = factor_df.columns.intersection(ref_factor.columns)

            if len(common_dates) == 0 or len(common_stocks) == 0:
                orthogonalized[factor_name] = factor_df.copy()
                continue

            # 对每个日期进行回归
            orthogonalized_df = factor_df.copy()

            for date in common_dates:
                y = factor_df.loc[date, common_stocks].values
                x = ref_factor.loc[date, common_stocks].values

                # 去除NaN
                mask = ~(np.isnan(y) | np.isnan(x))
                y_clean = y[mask]
                x_clean = x[mask]

                if len(y_clean) < 2:
                    continue

                # 线性回归
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

                # 计算残差
                y_pred = slope * x + intercept
                residual = y - y_pred

                orthogonalized_df.loc[date, common_stocks] = residual

            orthogonalized[factor_name] = orthogonalized_df

        return orthogonalized

    def _orthogonalize_gram_schmidt(self, reference_factor: str) -> Dict[str, pd.DataFrame]:
        """
        使用Gram-Schmidt正交化
        """
        # 简化实现：对每个因子减去其在参考因子方向上的投影
        orthogonalized = {}
        ref_factor = self.factors[reference_factor]

        for factor_name, factor_df in self.factors.items():
            if factor_name == reference_factor:
                orthogonalized[factor_name] = factor_df.copy()
                continue

            # 对齐数据
            common_dates = factor_df.index.intersection(ref_factor.index)
            common_stocks = factor_df.columns.intersection(ref_factor.columns)

            orthogonalized_df = factor_df.copy()

            for date in common_dates:
                f1 = factor_df.loc[date, common_stocks].values
                f2 = ref_factor.loc[date, common_stocks].values

                # 去除NaN
                mask = ~(np.isnan(f1) | np.isnan(f2))
                f1_clean = f1[mask]
                f2_clean = f2[mask]

                if len(f1_clean) == 0:
                    continue

                # 计算投影并减去
                dot_product = np.dot(f1_clean, f2_clean)
                norm_f2 = np.dot(f2_clean, f2_clean)

                if norm_f2 > 0:
                    projection = (dot_product / norm_f2) * f2
                    orthogonalized_df.loc[date, common_stocks] = f1 - projection

            orthogonalized[factor_name] = orthogonalized_df

        return orthogonalized

    def _orthogonalize_pca(self, reference_factor: str) -> Dict[str, pd.DataFrame]:
        """
        使用PCA进行因子正交化
        """
        # 将所有因子展平
        factor_list = []
        factor_names = []

        for name, df in self.factors.items():
            if name == reference_factor:
                continue
            factor_list.append(df.values.flatten())
            factor_names.append(name)

        if len(factor_list) == 0:
            return self.factors.copy()

        # 标准化
        scaler = StandardScaler()
        factor_matrix = np.array(factor_list).T
        factor_matrix_scaled = scaler.fit_transform(factor_matrix)

        # PCA
        pca = PCA(n_components=min(len(factor_list), len(factor_list[0])))
        pca_components = pca.fit_transform(factor_matrix_scaled)

        # 重建正交化后的因子
        orthogonalized = {}
        for i, name in enumerate(factor_names):
            original_shape = self.factors[name].shape
            orthogonalized_factor = pca_components[:, i].reshape(original_shape)
            orthogonalized[name] = pd.DataFrame(
                orthogonalized_factor,
                index=self.factors[name].index,
                columns=self.factors[name].columns
            )

        # 参考因子保持不变
        orthogonalized[reference_factor] = self.factors[reference_factor].copy()

        return orthogonalized

    def calculate_factor_correlation(self,
                                   method: str = 'pearson') -> pd.DataFrame:
        """
        计算因子之间的相关性

        Args:
            method: 相关系数方法 ('pearson' 或 'spearman')

        Returns:
            因子相关性矩阵DataFrame
        """
        # 将所有因子展平
        factor_series = {}

        for name, df in self.factors.items():
            factor_series[name] = df.stack()

        # 创建DataFrame
        factor_df = pd.DataFrame(factor_series)

        # 计算相关性
        if method == 'pearson':
            corr_matrix = factor_df.corr()
        elif method == 'spearman':
            corr_matrix = factor_df.corr(method='spearman')
        else:
            raise ValueError(f"Unknown method: {method}")

        return corr_matrix

    def select_best_factors(self,
                          return_df: pd.DataFrame,
                          top_n: int = 5,
                          metric: str = 'ic_mean') -> List[str]:
        """
        选择表现最好的因子

        Args:
            return_df: 收益率DataFrame
            top_n: 选择前几个因子
            metric: 选择指标 ('ic_mean', 'rank_ic_mean', 'ic_ir', 'rank_ic_ir')

        Returns:
            选择的因子名称列表
        """
        factor_scores = {}

        for factor_name, factor_df in self.factors.items():
            # 生成因子报告
            report = FactorMetrics.generate_report(
                factor_df,
                return_df,
                factor_name,
                periods=1
            )

            # 根据指标选择分数
            if metric == 'ic_mean':
                score = abs(report['ic_mean'])
            elif metric == 'rank_ic_mean':
                score = abs(report['rank_ic_mean'])
            elif metric == 'ic_ir':
                score = abs(report['ic_ir'])
            elif metric == 'rank_ic_ir':
                score = abs(report['rank_ic_ir'])
            else:
                raise ValueError(f"Unknown metric: {metric}")

            factor_scores[factor_name] = score

        # 排序并选择前n个
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, score in sorted_factors[:top_n]]

    def generate_combined_report(self,
                                 return_df: pd.DataFrame) -> pd.DataFrame:
        """
        生成多因子综合报告

        Args:
            return_df: 收益率DataFrame

        Returns:
            包含所有因子指标的DataFrame
        """
        reports = []

        for factor_name in self.factor_names:
            report = FactorMetrics.generate_report(
                self.factors[factor_name],
                return_df,
                factor_name,
                periods=1
            )
            reports.append(report)

        report_df = pd.DataFrame(reports)
        report_df = report_df.set_index('factor_name')

        return report_df

    @staticmethod
    def print_combined_report(report_df: pd.DataFrame):
        """
        打印多因子综合报告

        Args:
            report_df: generate_combined_report()生成的DataFrame
        """
        print("=" * 100)
        print("多因子综合分析报告")
        print("=" * 100)
        print()

        # 设置显示格式
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        print(report_df.to_string())
        print()

        print("=" * 100)

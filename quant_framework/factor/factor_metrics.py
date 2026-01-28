"""
因子指标计算模块
计算因子的IC、ICIR、Rank IC、Rank ICIR等指标
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats


class FactorMetrics:
    """
    因子指标计算类

    提供因子有效性评估的各种指标计算
    """

    @staticmethod
    def calculate_ic(factor_values: pd.Series,
                    returns: pd.Series,
                    method: str = 'pearson') -> float:
        """
        计算IC（Information Coefficient，信息系数）

        IC衡量因子值与未来收益率的相关性

        Args:
            factor_values: 因子值（Series，索引为日期或股票代码）
            returns: 未来收益率（Series，索引需与factor_values相同）
            method: 相关系数计算方法 ('pearson' 或 'spearman')

        Returns:
            IC值
        """
        # 确保数据是数值类型
        factor_values = pd.to_numeric(factor_values, errors='coerce')
        returns = pd.to_numeric(returns, errors='coerce')
        
        # 对齐数据
        aligned_data = pd.DataFrame({'factor': factor_values, 'return': returns}).dropna()

        if len(aligned_data) < 2:
            return np.nan

        if method == 'pearson':
            ic, _ = stats.pearsonr(aligned_data['factor'], aligned_data['return'])
        elif method == 'spearman':
            ic, _ = stats.spearmanr(aligned_data['factor'], aligned_data['return'])
        else:
            raise ValueError(f"Unknown method: {method}")

        return ic

    @staticmethod
    def calculate_rank_ic(factor_values: pd.Series,
                         returns: pd.Series) -> float:
        """
        计算Rank IC（秩相关系数）

        Rank IC是因子值排名与收益率排名的相关系数

        Args:
            factor_values: 因子值
            returns: 未来收益率

        Returns:
            Rank IC值
        """
        return FactorMetrics.calculate_ic(factor_values, returns, method='spearman')

    @staticmethod
    def calculate_icir(ic_series: pd.Series,
                       annualization: bool = True) -> float:
        """
        计算ICIR（Information Coefficient Information Ratio）

        ICIR = IC均值 / IC标准差

        Args:
            ic_series: IC时间序列
            annualization: 是否年化

        Returns:
            ICIR值
        """
        if len(ic_series) < 2:
            return np.nan

        mean_ic = ic_series.mean()
        std_ic = ic_series.std()

        if std_ic == 0:
            return np.nan

        icir = mean_ic / std_ic

        # 年化处理（假设一年有252个交易日）
        if annualization:
            icir *= np.sqrt(252)

        return icir

    @staticmethod
    def calculate_rank_icir(rank_ic_series: pd.Series,
                            annualization: bool = True) -> float:
        """
        计算Rank ICIR

        Args:
            rank_ic_series: Rank IC时间序列
            annualization: 是否年化

        Returns:
            Rank ICIR值
        """
        return FactorMetrics.calculate_icir(rank_ic_series, annualization)

    @staticmethod
    def calculate_ic_series(factor_df: pd.DataFrame,
                           return_df: pd.DataFrame,
                           periods: int = 1,
                           method: str = 'pearson') -> pd.Series:
        """
        计算IC时间序列

        Args:
            factor_df: 因子值DataFrame（索引为日期，列为股票代码）
            return_df: 收益率DataFrame（索引为日期，列为股票代码）
            periods: 预测期数（默认1，即预测下一期收益）
            method: 相关系数方法 ('pearson' 或 'spearman')

        Returns:
            IC时间序列（索引为日期，值为IC值）
        """
        ic_list = []

        # 获取共同的日期
        common_dates = factor_df.index.intersection(return_df.index)
        
        for i in range(len(common_dates) - periods):
            current_date = common_dates[i]
            future_date = common_dates[i + periods]

            # 获取当前日期的因子值
            current_factors = factor_df.loc[current_date]

            # 获取未来日期的收益率
            future_returns = return_df.loc[future_date]

            # 计算IC
            ic = FactorMetrics.calculate_ic(current_factors, future_returns, method)
            ic_list.append(ic)

        # 创建Series
        ic_series = pd.Series(ic_list, index=common_dates[:-periods])

        return ic_series

    @staticmethod
    def calculate_monotonicity(factor_values: pd.Series,
                               returns: pd.Series,
                               n_groups: int = 5) -> Dict:
        """
        计算因子单调性

        将因子值分为n组，计算各组的平均收益率，检验单调性

        Args:
            factor_values: 因子值
            returns: 收益率
            n_groups: 分组数

        Returns:
            包含分组收益和单调性检验结果的字典
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            'factor': factor_values,
            'return': returns
        }).dropna()

        if len(aligned_data) == 0:
            return {'group_returns': pd.Series(), 'monotonic': False}

        # 按因子值分组
        aligned_data['group'] = pd.qcut(aligned_data['factor'], q=n_groups, labels=False, duplicates='drop')

        # 计算各组平均收益
        group_returns = aligned_data.groupby('group')['return'].mean()

        # 检验单调性（简单判断：是否递增或递减）
        if len(group_returns) >= 2:
            monotonic = (group_returns.diff().dropna() > 0).all() or \
                       (group_returns.diff().dropna() < 0).all()
        else:
            monotonic = False

        return {
            'group_returns': group_returns,
            'monotonic': monotonic,
            'long_short_return': group_returns.iloc[-1] - group_returns.iloc[0]
        }

    @staticmethod
    def generate_report(factor_df: pd.DataFrame,
                       return_df: pd.DataFrame,
                       factor_name: str = 'Factor',
                       periods: int = 1) -> Dict:
        """
        生成因子分析报告

        Args:
            factor_df: 因子值DataFrame（索引为日期，列为股票代码）
            return_df: 收益率DataFrame（索引为日期，列为股票代码）
            factor_name: 因子名称
            periods: 预测期数

        Returns:
            包含各种指标的字典
        """
        report = {
            'factor_name': factor_name,
            'periods': periods
        }

        # 计算IC时间序列
        ic_series = FactorMetrics.calculate_ic_series(factor_df, return_df, periods, 'pearson')
        rank_ic_series = FactorMetrics.calculate_ic_series(factor_df, return_df, periods, 'spearman')

        # IC相关指标
        report['ic_mean'] = ic_series.mean()
        report['ic_std'] = ic_series.std()
        report['ic_ir'] = FactorMetrics.calculate_icir(ic_series, annualization=True)
        report['ic_positive_ratio'] = (ic_series > 0).sum() / len(ic_series)

        # Rank IC相关指标
        report['rank_ic_mean'] = rank_ic_series.mean()
        report['rank_ic_std'] = rank_ic_series.std()
        report['rank_ic_ir'] = FactorMetrics.calculate_icir(rank_ic_series, annualization=True)
        report['rank_ic_positive_ratio'] = (rank_ic_series > 0).sum() / len(rank_ic_series)

        # IC绝对值均值
        report['abs_ic_mean'] = ic_series.abs().mean()

        # 最大IC和最小IC
        report['max_ic'] = ic_series.max()
        report['min_ic'] = ic_series.min()

        # 样本数
        report['sample_size'] = len(ic_series)

        return report

    @staticmethod
    def print_report(report: Dict):
        """
        打印因子分析报告

        Args:
            report: generate_report()生成的报告字典
        """
        print("=" * 70)
        print(f"因子分析报告: {report['factor_name']}")
        print("=" * 70)
        print()

        print("IC指标（Pearson相关系数）:")
        print(f"  IC均值: {report['ic_mean']:.4f}")
        print(f"  IC标准差: {report['ic_std']:.4f}")
        print(f"  ICIR: {report['ic_ir']:.4f}")
        print(f"  IC>0占比: {report['ic_positive_ratio']:.2%}")
        print(f"  IC绝对值均值: {report['abs_ic_mean']:.4f}")
        print(f"  最大IC: {report['max_ic']:.4f}")
        print(f"  最小IC: {report['min_ic']:.4f}")
        print()

        print("Rank IC指标（Spearman秩相关系数）:")
        print(f"  Rank IC均值: {report['rank_ic_mean']:.4f}")
        print(f"  Rank IC标准差: {report['rank_ic_std']:.4f}")
        print(f"  Rank ICIR: {report['rank_ic_ir']:.4f}")
        print(f"  Rank IC>0占比: {report['rank_ic_positive_ratio']:.2%}")
        print()

        print("其他信息:")
        print(f"  预测期数: {report['periods']}")
        print(f"  样本数: {report['sample_size']}")
        print()

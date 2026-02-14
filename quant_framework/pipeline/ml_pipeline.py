"""
机器学习量化交易Pipeline
整合训练、评估、回测的完整流程
"""

from typing import Dict, Optional
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import gc

from ..data.data_handler_f import DataHandlerF
from ..strategy.ml_strategy import MLStrategy
from ..backtest.engine import BacktestEngine
from ..execution.transaction_cost import StandardCost
from ..performance.reports import ReportGenerator
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .data_loader import DataLoader
from ..utils.logger import get_logger

logger = get_logger(__name__)


def str_to_date(date_str: str) -> date:
    """将字符串转换为日期对象"""
    return datetime.strptime(date_str, '%Y-%m-%d').date()


class MLPipeline:
    """
    机器学习量化交易Pipeline

    完整流程：
    1. 训练阶段：训练LightGBM模型
    2. 评估阶段：计算IC、RankIC等因子有效性指标
    3. 回测阶段：使用模型进行回测，生成交易报告
    """

    def __init__(self, config: Dict):
        """
        初始化Pipeline

        Args:
            config: 配置字典，包含以下键：
                - data_path: 数据路径
                - factors: 因子列表（可选，默认使用所有alpha158因子）
                - train_start, train_end: 训练日期范围
                - valid_start, valid_end: 验证日期范围
                - test_start, test_end: 测试日期范围
                - backtest_start, backtest_end: 回测日期范围
                - model_params: 模型参数
                - strategy_params: 策略参数
                - backtest_params: 回测参数
        """
        self.config = self._validate_config(config)

        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path("examples/pipeline_outputs") / f"pipeline_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.model_trainer = None
        self.model_evaluator = None
        self.backtest_engine = None
        self.report_generator = ReportGenerator(
            output_dir=str(self.output_dir),
            create_timestamp_dir=False
        )

        # 结果存储
        self.model = None
        self.training_info = None
        self.evaluation_metrics = None
        self.backtest_results = None

        logger.info(f"✓ Pipeline初始化完成")
        logger.info(f"  - 输出目录: {self.output_dir}")

    def _validate_config(self, config: Dict) -> Dict:
        """验证配置并设置默认值"""
        # 设置默认值
        defaults = {
            'use_parquet': True,
            'num_workers': 4,
            'min_data_points': 50,
            'model_params': {
                'loss': 'mse',
                'num_boost_round': 1000,
                'early_stopping_rounds': 50,
            },
            'strategy_params': {
                'top_k': 10,
                'rebalance_days': 5,
                'stop_loss': 0.05,
                'stop_loss_check_daily': True,
            },
            'backtest_params': {
                'initial_capital': 1000000,
                'max_single_position_ratio': 1.0,
            },
        }

        # 合并配置
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and key in config:
                # 对于字典类型的默认值，合并而非覆盖
                for sub_key, sub_value in value.items():
                    if sub_key not in config[key]:
                        config[key][sub_key] = sub_value

        # 验证必需的配置项
        required_keys = [
            'data_path',
            'factors',  # 因子列表必须提供
            'train_start', 'train_end',
            'valid_start', 'valid_end',
            'test_start', 'test_end',
            'backtest_start', 'backtest_end',
        ]

        for key in required_keys:
            if key not in config:
                raise ValueError(f"缺少必需的配置项: {key}")

        return config

    def run_training(self) -> Dict:
        """
        阶段1：模型训练

        Returns:
            训练信息字典
        """
        logger.info("\n" + "=" * 60)
        logger.info("第2步：模型训练")
        logger.info("=" * 60)

        logger.info(f"使用因子数量: {len(self.config['factors'])}")

        # 创建单个 DataHandlerF 加载训练+验证集的所有数据
        # 这样可以避免重复加载相同的数据文件
        combined_start = str_to_date(self.config['train_start'])
        combined_end = str_to_date(self.config['valid_end'])

        logger.info(f"加载训练+验证集数据: {self.config['train_start']} 至 {self.config['valid_end']}")
        combined_data_handler = DataHandlerF(
            data_path=self.config['data_path'],
            min_data_points=self.config['min_data_points'],
            use_parquet=self.config.get('use_parquet', True),
            num_workers=self.config.get('num_workers', 4),
        )
        combined_data_handler.load_data(
            start_date=combined_start,
            end_date=combined_end,
            factors=self.config['factors'],
        )

        # 创建训练集 DataLoader（会从 DataHandlerF 中筛选训练日期范围的数据）
        train_loader = DataLoader(
            segment='train',
            data_handler=combined_data_handler,
            factors=self.config['factors'],
            start_date=self.config['train_start'],
            end_date=self.config['train_end'],
            label_threshold=self.config.get('label_threshold', 0.12),
            norm_method=self.config.get('norm_method', 'zscore')
        )

        # 创建验证集 DataLoader（会从同一个 DataHandlerF 中筛选验证日期范围的数据）
        valid_loader = DataLoader(
            segment='valid',
            data_handler=combined_data_handler,
            factors=self.config['factors'],
            start_date=self.config['valid_start'],
            end_date=self.config['valid_end'],
            label_threshold=self.config.get('label_threshold', 0.12),
            norm_method=self.config.get('norm_method', 'zscore')
        )

        # 创建模型训练器
        self.model_trainer = ModelTrainer(
            factors=self.config['factors'],
            model_params=self.config['model_params']
        )

        # 准备数据并训练
        self.model_trainer.prepare_data(train_loader, valid_loader)
        self.model = self.model_trainer.train()

        # 保存模型
        self.model_trainer.save_model(str(self.output_dir), "model")

        # 获取训练信息
        self.training_info = self.model_trainer.get_training_info()

        # ====== 内存清理 ======
        logger.info("\n清理训练阶段内存...")
        # 清理 DataLoader 中的数据
        if hasattr(train_loader, 'data'):
            train_loader.data = None
        if hasattr(valid_loader, 'data'):
            valid_loader.data = None
        train_loader = None
        valid_loader = None

        # 清理 ModelTrainer 中的 DataFrame
        if hasattr(self.model_trainer, 'train_df'):
            self.model_trainer.train_df = None
        if hasattr(self.model_trainer, 'valid_df'):
            self.model_trainer.valid_df = None

        # 清理 DataHandlerF
        combined_data_handler.all_data = None
        combined_data_handler = None

        # 强制垃圾回收
        gc.collect()
        logger.info("✓ 训练阶段内存已清理")

        return self.training_info

    def run_evaluation(self) -> Dict:
        """
        阶段2：模型评估

        Returns:
            评估指标字典
        """
        logger.info("\n" + "=" * 60)
        logger.info("第3步：模型评估")
        logger.info("=" * 60)

        if self.model is None:
            raise ValueError("模型尚未训练，请先调用run_training()")

        # 创建测试集 DataHandlerF
        test_data_handler = DataHandlerF(
            data_path=self.config['data_path'],
            min_data_points=self.config['min_data_points'],
            use_parquet=self.config.get('use_parquet', True),
            num_workers=self.config.get('num_workers', 4),
        )
        test_data_handler.load_data(
            start_date=str_to_date(self.config['test_start']),
            end_date=str_to_date(self.config['test_end'])
        )

        # 创建测试集 DataLoader
        test_loader = DataLoader(
            segment='test',
            data_handler=test_data_handler,
            factors=self.config['factors'],
            start_date=self.config['test_start'],
            end_date=self.config['test_end'],
            label_threshold=self.config.get('label_threshold', 0.12),
            norm_method=self.config.get('norm_method', 'zscore')
        )

        # 创建模型评估器
        self.model_evaluator = ModelEvaluator(model=self.model)

        # 进行预测
        self.model_evaluator.predict(test_loader)

        # 计算指标
        self.evaluation_metrics = self.model_evaluator.calculate_metrics()

        # 打印摘要
        logger.info("\n" + self.model_evaluator.get_evaluation_summary())

        # 保存报告
        report_path = self.model_evaluator.save_evaluation_report(str(self.output_dir))

        # ====== 内存清理 ======
        logger.info("\n清理评估阶段内存...")
        # 清理 DataLoader 中的数据
        if hasattr(test_loader, 'data'):
            test_loader.data = None
        test_loader = None

        # 清理 ModelEvaluator 中的数据
        if hasattr(self.model_evaluator, 'predictions'):
            self.model_evaluator.predictions = None
        if hasattr(self.model_evaluator, 'test_loader'):
            self.model_evaluator.test_loader = None

        # 清理 DataHandlerF
        test_data_handler.all_data = None
        test_data_handler = None

        # 强制垃圾回收
        gc.collect()
        logger.info("✓ 评估阶段内存已清理")

        return self.evaluation_metrics

    def run_backtest(self) -> Dict:
        """
        阶段3：回测

        Returns:
            回测结果字典
        """
        logger.info("\n" + "=" * 60)
        logger.info("第4步：回测")
        logger.info("=" * 60)

        if self.model is None:
            raise ValueError("模型尚未训练，请先调用run_training()")

        # 创建回测专用的 DataHandlerF
        backtest_data_handler = DataHandlerF(
            data_path=self.config['data_path'],
            min_data_points=self.config['min_data_points'],
            use_parquet=self.config.get('use_parquet', True),
            num_workers=self.config.get('num_workers', 4),
        )

        # 加载回测所需的数据
        backtest_data_handler.load_data(
            start_date=str_to_date(self.config['backtest_start']),
            end_date=str_to_date(self.config['backtest_end'])
        )

        # 创建策略
        strategy_params = self.config['strategy_params'].copy()
        strategy_params['model_path'] = str(self.output_dir / "model.pkl")

        strategy = MLStrategy(params=strategy_params)

        # 创建回测引擎
        self.backtest_engine = BacktestEngine(
            data_handler=backtest_data_handler,
            strategy=strategy,
            initial_capital=self.config['backtest_params']['initial_capital'],
            max_single_position_ratio=self.config['backtest_params'].get('max_single_position_ratio', 1.0),
            transaction_cost=StandardCost()
        )

        # 运行回测
        self.backtest_results = self.backtest_engine.run(
            start_date=self.config['backtest_start'],
            end_date=self.config['backtest_end'],
            verbose=True
        )

        # ====== 内存清理 ======
        logger.info("\n清理回测阶段内存...")
        # 注意：不能清理 backtest_engine 和 backtest_results，因为 save_results() 还需要使用
        # 只清理 DataHandlerF
        backtest_data_handler.all_data = None
        backtest_data_handler = None

        # 强制垃圾回收
        gc.collect()
        logger.info("✓ 回测阶段内存已清理")

        return self.backtest_results

    def save_results(self):
        """保存所有结果"""
        logger.info("\n" + "=" * 60)
        logger.info("第5步：保存结果")
        logger.info("=" * 60)

        if self.backtest_results is None:
            raise ValueError("回测尚未完成，请先调用run_backtest()")

        # 计算绩效指标
        from ..performance.analyzer import calculate_all_metrics

        metrics = calculate_all_metrics(
            portfolio_history=self.backtest_results['portfolio_history'],
            trades=self.backtest_results['trades'],
            initial_capital=self.backtest_results['initial_capital'],
            risk_free_rate=0.03
        )

        # 导出交易记录
        logger.info("\n导出交易记录...")
        self.report_generator.export_trades_to_csv(self.backtest_results['trades'])

        # 导出持仓历史
        logger.info("导出持仓历史...")
        self.report_generator.export_positions_to_csv(
            self.backtest_results['portfolio_history']
        )

        # 导出交易分析
        trade_analysis = self.backtest_results.get('trade_analysis')
        if trade_analysis:
            logger.info("导出交易分析...")
            self.report_generator.export_trade_analysis_to_csv(trade_analysis)

        # 导出绩效指标
        logger.info("导出绩效指标...")
        self.report_generator.export_metrics_to_json(
            metrics=metrics,
            trade_analysis=trade_analysis
        )

        # 绘制图表
        logger.info("绘制资金曲线...")
        self.report_generator.plot_equity_curve(
            portfolio_history=self.backtest_results['portfolio_history'],
            save=True,
            show=False
        )

        logger.info("\n" + "=" * 60)
        logger.info(f"✓ 所有结果已保存到: {self.output_dir}")
        logger.info("=" * 60)

    def run(self):
        """
        运行完整Pipeline

        Returns:
            包含所有结果的字典
        """
        logger.info("\n" + "=" * 70)
        logger.info("机器学习量化交易Pipeline")
        logger.info("=" * 70)

        # 运行三个阶段
        self.run_training()
        self.run_evaluation()
        self.run_backtest()
        self.save_results()

        # 返回所有结果
        return {
            'training_info': self.training_info,
            'evaluation_metrics': self.evaluation_metrics,
            'backtest_results': self.backtest_results,
            'output_dir': str(self.output_dir),
        }

    def get_results(self) -> Dict:
        """
        获取所有结果

        Returns:
            结果字典
        """
        return {
            'training_info': self.training_info,
            'evaluation_metrics': self.evaluation_metrics,
            'backtest_results': self.backtest_results,
            'output_dir': str(self.output_dir),
        }

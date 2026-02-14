"""
机器学习量化交易Pipeline使用示例
演示如何使用MLPipeline完成训练→评估→回测的完整流程
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant_framework import MLPipeline
from quant_framework.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """主函数"""
    
    factors = ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60']
    

    # Pipeline配置
    config = {
        # ==================== 因子配置 ====================
        'factors': factors,

        # ==================== 数据配置 ====================
        'data_path': '../data/factor/day/alpha158',
        'use_parquet': True,
        # 'num_workers': 32,
        'num_workers': 1,
        'min_data_points': 50,

        # ==================== 日期配置 ====================
        # 训练集：2005-2022年
        'train_start': '2005-05-01',
        'train_end': '2021-12-31',

        # 验证集：2022-2023年
        'valid_start': '2022-01-04',
        'valid_end': '2023-12-29',

        # 测试集：2024-2025年（用于评估IC等指标）
        'test_start': '2024-01-02',
        'test_end': '2025-12-30',

        # 回测集：2024年
        'backtest_start': '2024-01-02',
        'backtest_end': '2025-12-30',

        # ==================== 模型配置 ====================
        'model_params': {
            'loss': 'mse',  # 回归任务
            'num_boost_round': 1000,
            'early_stopping_rounds': 50,
            "colsample_bytree": 0.8879,
            "learning_rate": 0.01,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            # "max_depth": 12,
            "max_depth": 10,
            # "num_leaves": 510,
            "num_leaves": 255,
            "num_threads": 32,
        },

        # ==================== 策略配置 ====================
        'strategy_params': {
            'top_k': 3,  # 选取预测分数最高的10只股票
            'rebalance_days': 3,  # 每7个交易日调仓一次
            'weight_method': 'equal',  # 等权重
            'stop_loss': 0.01,  # 1%止损
            'stop_loss_check_daily': True,  # 每日检查止损
        },

        # ==================== 回测配置 ====================
        'backtest_params': {
            'initial_capital': 1000000,  # 初始资金100万
            'max_single_position_ratio': 1.0,  # 单只股票最大仓位100%
        },
    }

    logger.info("=" * 70)
    logger.info("机器学习量化交易Pipeline示例")
    logger.info("=" * 70)
    logger.info("")

    # 创建Pipeline
    pipeline = MLPipeline(config)

    # ==================== 方式1：一键运行完整流程 ====================
    # logger.info("\n【方式1】一键运行完整流程")
    # results = pipeline.run()

    # ==================== 方式2：分步运行 ====================
    logger.info("\n【方式2】分步运行")
    results = pipeline.run_training()      # 1. 训练模型
    # pipeline.run_evaluation()    # 2. 评估模型
    # pipeline.run_backtest()      # 3. 回测
    # pipeline.save_results()      # 4. 保存结果

    # ==================== 获取结果 ====================
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline执行完成！")
    logger.info("=" * 70)

    results = pipeline.get_results()

    logger.info(f"\n输出目录: {results['output_dir']}")

    logger.info("\n--- 训练信息 ---")
    training_info = results['training_info']
    logger.info(f"状态: {training_info['status']}")
    logger.info(f"训练样本数: {training_info.get('train_samples', 0):,}")
    logger.info(f"验证样本数: {training_info.get('valid_samples', 0):,}")

    logger.info("\n--- 评估指标 ---")
    eval_metrics = results['evaluation_metrics']
    logger.info(f"IC均值: {eval_metrics.get('IC_mean', 0):.4f}")
    logger.info(f"IC_IR: {eval_metrics.get('IC_IR', 0):.4f}")
    logger.info(f"RankIC均值: {eval_metrics.get('RankIC_mean', 0):.4f}")
    logger.info(f"RankIC_IR: {eval_metrics.get('RankIC_IR', 0):.4f}")

    logger.info("\n--- 回测结果 ---")
    backtest_results = results['backtest_results']
    logger.info(f"最终权益: {backtest_results['final_value']:,.2f} 元")
    logger.info(f"总收益率: {backtest_results['total_return']*100:.2f}%")

    trade_analysis = backtest_results.get('trade_analysis', {})
    if trade_analysis and trade_analysis.get('completed_trades', 0) > 0:
        logger.info(f"完整交易: {trade_analysis['completed_trades']} 笔")
        logger.info(f"胜率: {trade_analysis['win_rate']:.2f}%")
        logger.info(f"盈亏比: {trade_analysis['profit_loss_ratio']:.2f}")

    logger.info("\n" + "=" * 70)
    logger.info("所有报告和图表已保存到输出目录")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

"""
策略选股命令模块

提供策略选股预测功能。
对应 MLStrategy 类。
"""

import os
import argparse
import pandas as pd
from datetime import datetime
from quant_framework.cli.commands.base import BaseCommand
from quant_framework.utils.logger import get_logger

logger = get_logger(__name__)


class StrategyCommand(BaseCommand):
    """策略选股命令组"""

    @property
    def name(self) -> str:
        return "strategy"

    @property
    def help(self) -> str:
        return "策略选股命令：预测、选股"

    def _add_subcommands(self):
        """添加策略相关的子命令"""
        subparsers = self.parser.add_subparsers(
            dest='action',
            title='可用子命令',
            description='策略选股相关的子命令'
        )

        # predict 子命令
        predict_parser = subparsers.add_parser(
            'predict',
            help='预测选股',
            description='使用机器学习模型预测Top N股票'
        )
        predict_parser.add_argument(
            '--db-path',
            type=str,
            required=True,
            help='数据库路径'
        )
        predict_parser.add_argument(
            '--model-path',
            type=str,
            required=True,
            help='模型文件路径'
        )
        predict_parser.add_argument(
            '--date',
            type=str,
            help='预测日期（格式：YYYY-MM-DD），默认为最新交易日'
        )
        predict_parser.add_argument(
            '--top-n',
            type=int,
            default=20,
            help='选择前N只股票（默认：20）'
        )
        predict_parser.add_argument(
            '--output',
            type=str,
            help='输出文件路径（默认：signals/YYYYMMDD.csv）'
        )
        predict_parser.set_defaults(func=self.predict)

    def predict(self, args: argparse.Namespace):
        """
        执行预测选股

        Args:
            args: 命令行参数
        """
        logger.info("开始执行预测选股...")
        logger.info(f"数据库路径: {args.db_path}")
        logger.info(f"模型路径: {args.model_path}")
        logger.info(f"Top N: {args.top_n}")

        from quant_framework import DataHandler, DataUpdater, MLStrategy

        # 初始化 DataHandler
        data_handler = DataHandler(args.db_path)

        # 确定预测日期
        if args.date:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        else:
            data_updater = DataUpdater(data_handler)
            target_date = data_updater.get_appropriate_end_date()
            logger.info(f"未指定日期，使用最新交易日: {target_date}")

        logger.info(f"预测日期: {target_date}")

        # 创建 MLStrategy 实例
        strategy = MLStrategy({
            'model_path': args.model_path,
        })

        # 执行预测
        predictions = strategy.realtime_prediction(data_handler, target_date, args.top_n)

        if predictions is None:
            logger.warning("预测结果为空")
            return

        # 处理预测结果
        rank_list, code_list, name_list, score_list = [], [], [], []

        for prediction in predictions:
            pred_date = prediction['date']
            pred_result = prediction['predictions']

            if pred_date != target_date:
                continue

            rank = 1
            for stock_code, pred_score in pred_result:
                # 获取股票名称
                try:
                    stock_info = data_handler.get_stock_info(int(stock_code))
                    if stock_info is None or len(stock_info) == 0:
                        stock_name = "股票名称获取失败"
                    else:
                        stock_name = stock_info['stock_name'].iloc[0]
                except Exception:
                    stock_name = "股票名称获取失败"

                logger.info(f"rank={rank}, code={stock_code}, name={stock_name}, score={pred_score}")
                rank_list.append(rank)
                code_list.append(stock_code)
                name_list.append(stock_name)
                score_list.append(pred_score)

                rank += 1

        # 创建结果 DataFrame
        pred_df = pd.DataFrame({
            'rank': rank_list,
            'code': code_list,
            'name': name_list,
            'score': score_list
        })

        # 确定输出路径
        if args.output:
            pred_save_path = args.output
        else:
            # 从模型路径提取模型名称
            model_filename = os.path.basename(args.model_path)
            model_name = os.path.splitext(model_filename)[0]
            pred_save_dir = os.path.join(os.getcwd(), 'signals')
            pred_save_path = os.path.join(pred_save_dir, f'{target_date.strftime("%Y%m%d")}_{model_name}_top{args.top_n}.csv')

        # 确保输出目录存在
        pred_save_dir = os.path.dirname(pred_save_path)
        if pred_save_dir and not os.path.exists(pred_save_dir):
            os.makedirs(pred_save_dir)

        # 保存结果
        pred_df.to_csv(pred_save_path, index=False)
        logger.info(f"{target_date} 的预测结果已保存到 {pred_save_path}")
        logger.info(f"共选出 {len(pred_df)} 只股票")

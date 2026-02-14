"""
报告生成模块

用于生成选股预测结果的各种格式报告。

Usage:
    from quant_framework.reporting import ReportGenerator, StockPickReport

    generator = ReportGenerator()

    # 从CSV加载
    report = generator.load_from_csv("signals/20260212_top50.csv")

    # 生成HTML报告
    html = generator.to_html(report)

    # 生成Markdown报告
    md = generator.to_markdown(report)
"""

from .report_generator import ReportGenerator, StockPickReport

__all__ = [
    'ReportGenerator',
    'StockPickReport',
]

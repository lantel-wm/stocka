"""
报告生成器模块

读取选股预测结果，生成美观的HTML/Markdown报告。
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class StockPickReport:
    """选股报告数据类"""

    def __init__(self, date_str: str, stocks: List[Dict[str, Any]], model_name: str = ""):
        """
        Args:
            date_str: 日期字符串 (YYYY-MM-DD 或 YYYYMMDD)
            stocks: 股票列表，每项包含 rank, code, name, score
            model_name: 模型名称
        """
        self.date_str = self._normalize_date(date_str)
        self.stocks = stocks
        self.model_name = model_name
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _normalize_date(self, date_str: str) -> str:
        """统一日期格式为 YYYY-MM-DD"""
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return date_str

    @property
    def count(self) -> int:
        """股票数量"""
        return len(self.stocks)

    @property
    def top_score(self) -> float:
        """最高分"""
        return self.stocks[0].get("score", 0) if self.stocks else 0

    @property
    def avg_score(self) -> float:
        """平均分"""
        if not self.stocks:
            return 0
        return sum(s.get("score", 0) for s in self.stocks) / len(self.stocks)


class ReportGenerator:
    """选股报告生成器

    将CSV预测结果转换为美观的HTML或Markdown格式。

    Example:
        >>> generator = ReportGenerator()
        >>> report = generator.load_from_csv("signals/20260212_top50.csv")
        >>> html = generator.to_html(report)
        >>> print(html)
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Args:
            template_dir: 模板目录路径，默认使用内置模板
        """
        self.template_dir = Path(template_dir) if template_dir else None

    def load_from_csv(self, csv_path: str, model_name: str = "") -> StockPickReport:
        """从CSV文件加载预测结果

        Args:
            csv_path: CSV文件路径
            model_name: 模型名称

        Returns:
            StockPickReport: 报告数据对象
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

        # 从文件名提取日期
        date_str = self._extract_date_from_filename(path.name)

        # 读取CSV
        df = pd.read_csv(csv_path)

        # 转换为字典列表
        stocks = df.to_dict("records")

        logger.info(f"加载预测结果: {csv_path}, 共 {len(stocks)} 只股票")
        return StockPickReport(date_str, stocks, model_name)

    def _extract_date_from_filename(self, filename: str) -> str:
        """从文件名提取日期"""
        # 尝试匹配 YYYYMMDD 格式
        import re
        match = re.search(r"(\d{8})", filename)
        if match:
            date_str = match.group(1)
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return datetime.now().strftime("%Y-%m-%d")

    def find_latest_report(self, signals_dir: str = "signals", pattern: str = "*.csv") -> Optional[str]:
        """查找最新的预测报告文件

        Args:
            signals_dir: signals目录路径
            pattern: 文件匹配模式

        Returns:
            str: 最新文件路径，如果没有找到返回None
        """
        dir_path = Path(signals_dir)
        if not dir_path.exists():
            logger.warning(f"Signals目录不存在: {signals_dir}")
            return None

        csv_files = list(dir_path.glob(pattern))
        if not csv_files:
            logger.warning(f"目录中没有找到CSV文件: {signals_dir}")
            return None

        # 按修改时间排序
        latest = max(csv_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"找到最新报告: {latest}")
        return str(latest)

    def to_html(self, report: StockPickReport, style: str = "detailed") -> str:
        """生成HTML格式报告

        Args:
            report: 报告数据对象
            style: 模板风格 (simple/detailed)

        Returns:
            str: HTML内容
        """
        if style == "simple":
            return self._generate_simple_html(report)
        return self._generate_detailed_html(report)

    def _generate_simple_html(self, report: StockPickReport) -> str:
        """生成简洁版HTML报告"""
        rows = ""
        for stock in report.stocks[:10]:  # 只显示前10
            rows += f"""
                <tr>
                    <td style="padding:8px;border-bottom:1px solid #eee;text-align:center;">{stock.get("rank", "")}</td>
                    <td style="padding:8px;border-bottom:1px solid #eee;">{stock.get("code", "")}</td>
                    <td style="padding:8px;border-bottom:1px solid #eee;font-weight:bold;">{stock.get("name", "")}</td>
                    <td style="padding:8px;border-bottom:1px solid #eee;text-align:right;color:#1890ff;">{stock.get("score", 0):.4f}</td>
                </tr>
            """

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>选股日报 - {report.date_str}</title>
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.6;color:#333;max-width:600px;margin:0 auto;padding:20px;">
    <h2 style="color:#1890ff;border-bottom:2px solid #1890ff;padding-bottom:10px;">📈 每日选股推荐</h2>
    <p style="color:#666;">日期：<strong>{report.date_str}</strong> | 共 <strong>{report.count}</strong> 只股票</p>

    <table style="width:100%;border-collapse:collapse;margin-top:20px;font-size:14px;">
        <thead>
            <tr style="background:#f5f5f5;">
                <th style="padding:10px;text-align:center;">排名</th>
                <th style="padding:10px;text-align:left;">代码</th>
                <th style="padding:10px;text-align:left;">名称</th>
                <th style="padding:10px;text-align:right;">得分</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>

    <p style="color:#999;font-size:12px;margin-top:30px;text-align:center;">
        由 Stocka 量化框架自动生成 | {report.created_at}
    </p>
</body>
</html>"""

    def _generate_detailed_html(self, report: StockPickReport) -> str:
        """生成详细版HTML报告"""
        # 生成所有行
        rows = ""
        for i, stock in enumerate(report.stocks):
            rank = stock.get("rank", i + 1)
            code = stock.get("code", "")
            name = stock.get("name", "")
            score = stock.get("score", 0)

            # 根据排名设置颜色
            if rank == 1:
                rank_color = "#ff4d4f"
                bg_color = "#fff1f0"
            elif rank <= 3:
                rank_color = "#fa8c16"
                bg_color = "#fff7e6"
            elif rank <= 10:
                rank_color = "#1890ff"
                bg_color = "#e6f7ff"
            else:
                rank_color = "#666"
                bg_color = "transparent"

            rows += f"""
                <tr style="background:{bg_color};">
                    <td style="padding:10px;border-bottom:1px solid #f0f0f0;text-align:center;font-weight:bold;color:{rank_color};">{rank}</td>
                    <td style="padding:10px;border-bottom:1px solid #f0f0f0;font-family:monospace;">{code}</td>
                    <td style="padding:10px;border-bottom:1px solid #f0f0f0;font-weight:500;">{name}</td>
                    <td style="padding:10px;border-bottom:1px solid #f0f0f0;text-align:right;font-family:monospace;color:#1890ff;">{score:.4f}</td>
                </tr>
            """

        # 统计信息
        top_10_avg = sum(s.get("score", 0) for s in report.stocks[:10]) / 10 if report.stocks else 0
        top_20_avg = sum(s.get("score", 0) for s in report.stocks[:20]) / 20 if len(report.stocks) >= 20 else top_10_avg

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>选股日报 - {report.date_str}</title>
    <style>
        @media only screen and (max-width: 600px) {{
            .container {{ padding: 10px !important; }}
            .stats {{ flex-direction: column !important; }}
            .stat-item {{ width: 100% !important; margin-bottom: 10px; }}
        }}
    </style>
</head>
<body style="margin:0;padding:0;background:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;">
    <div class="container" style="max-width:700px;margin:0 auto;padding:20px;">
        <!-- 头部 -->
        <div style="background:linear-gradient(135deg,#1890ff 0%,#36cfc9 100%);padding:30px;border-radius:8px 8px 0 0;color:white;text-align:center;">
            <h1 style="margin:0;font-size:24px;">📈 每日选股推荐</h1>
            <p style="margin:10px 0 0 0;opacity:0.9;">{report.date_str} | 基于机器学习模型预测</p>
        </div>

        <!-- 统计卡片 -->
        <div class="stats" style="background:white;padding:20px;display:flex;justify-content:space-around;border-bottom:1px solid #f0f0f0;">
            <div class="stat-item" style="text-align:center;flex:1;">
                <div style="font-size:28px;font-weight:bold;color:#1890ff;">{report.count}</div>
                <div style="font-size:12px;color:#999;">推荐股票数</div>
            </div>
            <div class="stat-item" style="text-align:center;flex:1;">
                <div style="font-size:28px;font-weight:bold;color:#52c41a;">{report.top_score:.4f}</div>
                <div style="font-size:12px;color:#999;">最高得分</div>
            </div>
            <div class="stat-item" style="text-align:center;flex:1;">
                <div style="font-size:28px;font-weight:bold;color:#fa8c16;">{report.avg_score:.4f}</div>
                <div style="font-size:12px;color:#999;">平均得分</div>
            </div>
            <div class="stat-item" style="text-align:center;flex:1;">
                <div style="font-size:28px;font-weight:bold;color:#722ed1;">{top_10_avg:.4f}</div>
                <div style="font-size:12px;color:#999;">Top10均分</div>
            </div>
        </div>

        <!-- 股票列表 -->
        <div style="background:white;padding:20px;border-radius:0 0 8px 8px;">
            <h3 style="margin:0 0 15px 0;color:#333;font-size:16px;">🎯 推荐列表 (Top {report.count})</h3>
            <table style="width:100%;border-collapse:collapse;font-size:14px;">
                <thead>
                    <tr style="background:#fafafa;">
                        <th style="padding:12px 10px;text-align:center;color:#666;font-weight:600;border-bottom:2px solid #f0f0f0;width:60px;">排名</th>
                        <th style="padding:12px 10px;text-align:left;color:#666;font-weight:600;border-bottom:2px solid #f0f0f0;">股票代码</th>
                        <th style="padding:12px 10px;text-align:left;color:#666;font-weight:600;border-bottom:2px solid #f0f0f0;">股票名称</th>
                        <th style="padding:12px 10px;text-align:right;color:#666;font-weight:600;border-bottom:2px solid #f0f0f0;width:100px;">预测得分</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>

        <!-- 底部 -->
        <div style="text-align:center;padding:20px;color:#999;font-size:12px;">
            <p style="margin:0;">由 Stocka 量化框架自动生成</p>
            <p style="margin:5px 0 0 0;">{report.created_at}</p>
            <p style="margin:5px 0 0 0;font-size:11px;color:#bbb;">⚠️ 仅供参考，不构成投资建议</p>
        </div>
    </div>
</body>
</html>"""

    def to_markdown(self, report: StockPickReport) -> str:
        """生成Markdown格式报告"""
        lines = [
            f"# 📈 每日选股推荐 - {report.date_str}",
            "",
            "## 统计信息",
            "",
            f"- **推荐股票数**: {report.count}",
            f"- **最高得分**: {report.top_score:.4f}",
            f"- **平均得分**: {report.avg_score:.4f}",
            "",
            "## 推荐列表",
            "",
            "| 排名 | 代码 | 名称 | 得分 |",
            "|------|------|------|------|",
        ]

        for stock in report.stocks:
            rank = stock.get("rank", "")
            code = stock.get("code", "")
            name = stock.get("name", "")
            score = stock.get("score", 0)
            lines.append(f"| {rank} | {code} | {name} | {score:.4f} |")

        lines.extend([
            "",
            "---",
            "",
            f"*由 Stocka 量化框架自动生成于 {report.created_at}*",
            "",
            "⚠️ **免责声明**: 以上内容仅供参考，不构成投资建议。",
        ])

        return "\n".join(lines)

    def generate_summary(self, report: StockPickReport) -> str:
        """生成纯文本摘要（用于日志或简单通知）"""
        top5 = report.stocks[:5]
        stocks_text = "\n".join([
            f"  {s.get('rank', i+1)}. {s.get('code', '')} {s.get('name', '')} (得分: {s.get('score', 0):.4f})"
            for i, s in enumerate(top5)
        ])

        return f"""【选股日报】{report.date_str}

共推荐 {report.count} 只股票，Top 5 如下:
{stocks_text}

最高得分: {report.top_score:.4f}
平均得分: {report.avg_score:.4f}

⚠️ 仅供参考，不构成投资建议
"""

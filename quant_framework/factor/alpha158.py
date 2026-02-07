from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

from .base_factor import BaseFactor

class Alpha158(BaseFactor):
    """
    Alpha158 因子库

    一次性计算所有Alpha158因子，包括：
    - KBar 因子: KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
    - Price 因子: OPEN0, HIGH0, LOW0, VWAP0
    - Volume 因子: VOLUME0
    - Rolling 因子: ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD,
      RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD,
      VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD
      (每个因子5个窗口: 5, 10, 20, 30, 60)
    """
    
        # Alpha158 因子定义
        # 格式: (factor_id, factor_name, factor_category, factor_desc)
    DEFINITIONS = [
        # KBar 因子 (ID: 1-9)
        (1, 'KMID', 'alpha158', '收盘价与开盘价的差值除以收盘价'),
        (2, 'KLEN', 'alpha158', '最高价与最低价的差值除以开盘价'),
        (3, 'KMID2', 'alpha158', '收盘价与开盘价的差值除以振幅'),
        (4, 'KUP', 'alpha158', '最高价与（开盘价、收盘价较大值）的差值除以开盘价'),
        (5, 'KUP2', 'alpha158', '最高价与（开盘价、收盘价较大值）的差值除以振幅'),
        (6, 'KLOW', 'alpha158', '（开盘价、收盘价较小值）与最低价的差值除以开盘价'),
        (7, 'KLOW2', 'alpha158', '（开盘价、收盘价较小值）与最低价的差值除以振幅'),
        (8, 'KSFT', 'alpha158', '2倍收盘价减去最高价和最低价的差值除以开盘价'),
        (9, 'KSFT2', 'alpha158', '2倍收盘价减去最高价和最低价的差值除以振幅'),

        # Price 因子 (ID: 10-12)
        (10, 'OPEN0', 'alpha158', '开盘价除以收盘价'),
        (11, 'HIGH0', 'alpha158', '最高价除以收盘价'),
        (12, 'LOW0', 'alpha158', '最低价除以收盘价'),

        # ROC 因子 (ID: 13-17, 5个窗口)
        (13, 'ROC5', 'alpha158', '5日收益率'),
        (14, 'ROC10', 'alpha158', '10日收益率'),
        (15, 'ROC20', 'alpha158', '20日收益率'),
        (16, 'ROC30', 'alpha158', '30日收益率'),
        (17, 'ROC60', 'alpha158', '60日收益率'),

        # MA 因子 (ID: 18-22, 5个窗口)
        (18, 'MA5', 'alpha158', '5日移动平均除以收盘价'),
        (19, 'MA10', 'alpha158', '10日移动平均除以收盘价'),
        (20, 'MA20', 'alpha158', '20日移动平均除以收盘价'),
        (21, 'MA30', 'alpha158', '30日移动平均除以收盘价'),
        (22, 'MA60', 'alpha158', '60日移动平均除以收盘价'),

        # STD 因子 (ID: 23-27, 5个窗口)
        (23, 'STD5', 'alpha158', '5日标准差除以收盘价'),
        (24, 'STD10', 'alpha158', '10日标准差除以收盘价'),
        (25, 'STD20', 'alpha158', '20日标准差除以收盘价'),
        (26, 'STD30', 'alpha158', '30日标准差除以收盘价'),
        (27, 'STD60', 'alpha158', '60日标准差除以收盘价'),

        # BETA 因子 (ID: 28-32, 5个窗口)
        (28, 'BETA5', 'alpha158', '5日线性回归斜率除以收盘价'),
        (29, 'BETA10', 'alpha158', '10日线性回归斜率除以收盘价'),
        (30, 'BETA20', 'alpha158', '20日线性回归斜率除以收盘价'),
        (31, 'BETA30', 'alpha158', '30日线性回归斜率除以收盘价'),
        (32, 'BETA60', 'alpha158', '60日线性回归斜率除以收盘价'),

        # RSQR 因子 (ID: 33-37, 5个窗口)
        (33, 'RSQR5', 'alpha158', '5日线性回归R平方'),
        (34, 'RSQR10', 'alpha158', '10日线性回归R平方'),
        (35, 'RSQR20', 'alpha158', '20日线性回归R平方'),
        (36, 'RSQR30', 'alpha158', '30日线性回归R平方'),
        (37, 'RSQR60', 'alpha158', '60日线性回归R平方'),

        # RESI 因子 (ID: 38-42, 5个窗口)
        (38, 'RESI5', 'alpha158', '5日线性回归残差除以收盘价'),
        (39, 'RESI10', 'alpha158', '10日线性回归残差除以收盘价'),
        (40, 'RESI20', 'alpha158', '20日线性回归残差除以收盘价'),
        (41, 'RESI30', 'alpha158', '30日线性回归残差除以收盘价'),
        (42, 'RESI60', 'alpha158', '60日线性回归残差除以收盘价'),

        # MAX 因子 (ID: 43-47, 5个窗口)
        (43, 'MAX5', 'alpha158', '5日最高价除以收盘价'),
        (44, 'MAX10', 'alpha158', '10日最高价除以收盘价'),
        (45, 'MAX20', 'alpha158', '20日最高价除以收盘价'),
        (46, 'MAX30', 'alpha158', '30日最高价除以收盘价'),
        (47, 'MAX60', 'alpha158', '60日最高价除以收盘价'),

        # MIN 因子 (ID: 48-52, 5个窗口)
        (48, 'MIN5', 'alpha158', '5日最低价除以收盘价'),
        (49, 'MIN10', 'alpha158', '10日最低价除以收盘价'),
        (50, 'MIN20', 'alpha158', '20日最低价除以收盘价'),
        (51, 'MIN30', 'alpha158', '30日最低价除以收盘价'),
        (52, 'MIN60', 'alpha158', '60日最低价除以收盘价'),

        # QTLU 因子 (ID: 53-57, 5个窗口)
        (53, 'QTLU5', 'alpha158', '5日80%分位数除以收盘价'),
        (54, 'QTLU10', 'alpha158', '10日80%分位数除以收盘价'),
        (55, 'QTLU20', 'alpha158', '20日80%分位数除以收盘价'),
        (56, 'QTLU30', 'alpha158', '30日80%分位数除以收盘价'),
        (57, 'QTLU60', 'alpha158', '60日80%分位数除以收盘价'),

        # QTLD 因子 (ID: 58-62, 5个窗口)
        (58, 'QTLD5', 'alpha158', '5日20%分位数除以收盘价'),
        (59, 'QTLD10', 'alpha158', '10日20%分位数除以收盘价'),
        (60, 'QTLD20', 'alpha158', '20日20%分位数除以收盘价'),
        (61, 'QTLD30', 'alpha158', '30日20%分位数除以收盘价'),
        (62, 'QTLD60', 'alpha158', '60日20%分位数除以收盘价'),

        # RSV 因子 (ID: 63-67, 5个窗口)
        (63, 'RSV5', 'alpha158', '5日RSV'),
        (64, 'RSV10', 'alpha158', '10日RSV'),
        (65, 'RSV20', 'alpha158', '20日RSV'),
        (66, 'RSV30', 'alpha158', '30日RSV'),
        (67, 'RSV60', 'alpha158', '60日RSV'),

        # IMAX 因子 (ID: 68-72, 5个窗口)
        (68, 'IMAX5', 'alpha158', '5日内最高价位置除以窗口长度'),
        (69, 'IMAX10', 'alpha158', '10日内最高价位置除以窗口长度'),
        (70, 'IMAX20', 'alpha158', '20日内最高价位置除以窗口长度'),
        (71, 'IMAX30', 'alpha158', '30日内最高价位置除以窗口长度'),
        (72, 'IMAX60', 'alpha158', '60日内最高价位置除以窗口长度'),

        # IMIN 因子 (ID: 73-77, 5个窗口)
        (73, 'IMIN5', 'alpha158', '5日内最低价位置除以窗口长度'),
        (74, 'IMIN10', 'alpha158', '10日内最低价位置除以窗口长度'),
        (75, 'IMIN20', 'alpha158', '20日内最低价位置除以窗口长度'),
        (76, 'IMIN30', 'alpha158', '30日内最低价位置除以窗口长度'),
        (77, 'IMIN60', 'alpha158', '60日内最低价位置除以窗口长度'),

        # IMXD 因子 (ID: 78-82, 5个窗口)
        (78, 'IMXD5', 'alpha158', '5日内最高价与最低价位置之差除以窗口长度'),
        (79, 'IMXD10', 'alpha158', '10日内最高价与最低价位置之差除以窗口长度'),
        (80, 'IMXD20', 'alpha158', '20日内最高价与最低价位置之差除以窗口长度'),
        (81, 'IMXD30', 'alpha158', '30日内最高价与最低价位置之差除以窗口长度'),
        (82, 'IMXD60', 'alpha158', '60日内最高价与最低价位置之差除以窗口长度'),

        # CORR 因子 (ID: 83-87, 5个窗口)
        (83, 'CORR5', 'alpha158', '5日收盘价与成交量的相关系数'),
        (84, 'CORR10', 'alpha158', '10日收盘价与成交量的相关系数'),
        (85, 'CORR20', 'alpha158', '20日收盘价与成交量的相关系数'),
        (86, 'CORR30', 'alpha158', '30日收盘价与成交量的相关系数'),
        (87, 'CORR60', 'alpha158', '60日收盘价与成交量的相关系数'),

        # CORD 因子 (ID: 88-92, 5个窗口)
        (88, 'CORD5', 'alpha158', '5日价格变化率与成交量变化率的相关系数'),
        (89, 'CORD10', 'alpha158', '10日价格变化率与成交量变化率的相关系数'),
        (90, 'CORD20', 'alpha158', '20日价格变化率与成交量变化率的相关系数'),
        (91, 'CORD30', 'alpha158', '30日价格变化率与成交量变化率的相关系数'),
        (92, 'CORD60', 'alpha158', '60日价格变化率与成交量变化率的相关系数'),

        # CNTP 因子 (ID: 93-97, 5个窗口)
        (93, 'CNTP5', 'alpha158', '5日内上涨天数占比'),
        (94, 'CNTP10', 'alpha158', '10日内上涨天数占比'),
        (95, 'CNTP20', 'alpha158', '20日内上涨天数占比'),
        (96, 'CNTP30', 'alpha158', '30日内上涨天数占比'),
        (97, 'CNTP60', 'alpha158', '60日内上涨天数占比'),

        # CNTN 因子 (ID: 98-102, 5个窗口)
        (98, 'CNTN5', 'alpha158', '5日内下跌天数占比'),
        (99, 'CNTN10', 'alpha158', '10日内下跌天数占比'),
        (100, 'CNTN20', 'alpha158', '20日内下跌天数占比'),
        (101, 'CNTN30', 'alpha158', '30日内下跌天数占比'),
        (102, 'CNTN60', 'alpha158', '60日内下跌天数占比'),

        # CNTD 因子 (ID: 103-107, 5个窗口)
        (103, 'CNTD5', 'alpha158', '5日内上涨天数占比与下跌天数占比之差'),
        (104, 'CNTD10', 'alpha158', '10日内上涨天数占比与下跌天数占比之差'),
        (105, 'CNTD20', 'alpha158', '20日内上涨天数占比与下跌天数占比之差'),
        (106, 'CNTD30', 'alpha158', '30日内上涨天数占比与下跌天数占比之差'),
        (107, 'CNTD60', 'alpha158', '60日内上涨天数占比与下跌天数占比之差'),

        # SUMP 因子 (ID: 108-112, 5个窗口)
        (108, 'SUMP5', 'alpha158', '5日内上涨幅度之和除以总变化幅度'),
        (109, 'SUMP10', 'alpha158', '10日内上涨幅度之和除以总变化幅度'),
        (110, 'SUMP20', 'alpha158', '20日内上涨幅度之和除以总变化幅度'),
        (111, 'SUMP30', 'alpha158', '30日内上涨幅度之和除以总变化幅度'),
        (112, 'SUMP60', 'alpha158', '60日内上涨幅度之和除以总变化幅度'),

        # SUMN 因子 (ID: 113-117, 5个窗口)
        (113, 'SUMN5', 'alpha158', '5日内下跌幅度之和除以总变化幅度'),
        (114, 'SUMN10', 'alpha158', '10日内下跌幅度之和除以总变化幅度'),
        (115, 'SUMN20', 'alpha158', '20日内下跌幅度之和除以总变化幅度'),
        (116, 'SUMN30', 'alpha158', '30日内下跌幅度之和除以总变化幅度'),
        (117, 'SUMN60', 'alpha158', '60日内下跌幅度之和除以总变化幅度'),

        # SUMD 因子 (ID: 118-122, 5个窗口)
        (118, 'SUMD5', 'alpha158', '5日内上涨与下跌幅度之差除以总变化幅度'),
        (119, 'SUMD10', 'alpha158', '10日内上涨与下跌幅度之差除以总变化幅度'),
        (120, 'SUMD20', 'alpha158', '20日内上涨与下跌幅度之差除以总变化幅度'),
        (121, 'SUMD30', 'alpha158', '30日内上涨与下跌幅度之差除以总变化幅度'),
        (122, 'SUMD60', 'alpha158', '60日内上涨与下跌幅度之差除以总变化幅度'),

        # VMA 因子 (ID: 123-127, 5个窗口)
        (123, 'VMA5', 'alpha158', '5日平均成交量除以当日成交量'),
        (124, 'VMA10', 'alpha158', '10日平均成交量除以当日成交量'),
        (125, 'VMA20', 'alpha158', '20日平均成交量除以当日成交量'),
        (126, 'VMA30', 'alpha158', '30日平均成交量除以当日成交量'),
        (127, 'VMA60', 'alpha158', '60日平均成交量除以当日成交量'),

        # VSTD 因子 (ID: 128-132, 5个窗口)
        (128, 'VSTD5', 'alpha158', '5日成交量标准差除以当日成交量'),
        (129, 'VSTD10', 'alpha158', '10日成交量标准差除以当日成交量'),
        (130, 'VSTD20', 'alpha158', '20日成交量标准差除以当日成交量'),
        (131, 'VSTD30', 'alpha158', '30日成交量标准差除以当日成交量'),
        (132, 'VSTD60', 'alpha158', '60日成交量标准差除以当日成交量'),

        # WVMA 因子 (ID: 133-137, 5个窗口)
        (133, 'WVMA5', 'alpha158', '5日加权成交量标准差除以加权平均成交量'),
        (134, 'WVMA10', 'alpha158', '10日加权成交量标准差除以加权平均成交量'),
        (135, 'WVMA20', 'alpha158', '20日加权成交量标准差除以加权平均成交量'),
        (136, 'WVMA30', 'alpha158', '30日加权成交量标准差除以加权平均成交量'),
        (137, 'WVMA60', 'alpha158', '60日加权成交量标准差除以加权平均成交量'),

        # VSUMP 因子 (ID: 138-142, 5个窗口)
        (138, 'VSUMP5', 'alpha158', '5日内成交量上涨之和除以总变化'),
        (139, 'VSUMP10', 'alpha158', '10日内成交量上涨之和除以总变化'),
        (140, 'VSUMP20', 'alpha158', '20日内成交量上涨之和除以总变化'),
        (141, 'VSUMP30', 'alpha158', '30日内成交量上涨之和除以总变化'),
        (142, 'VSUMP60', 'alpha158', '60日内成交量上涨之和除以总变化'),

        # VSUMN 因子 (ID: 143-147, 5个窗口)
        (143, 'VSUMN5', 'alpha158', '5日内成交量下跌之和除以总变化'),
        (144, 'VSUMN10', 'alpha158', '10日内成交量下跌之和除以总变化'),
        (145, 'VSUMN20', 'alpha158', '20日内成交量下跌之和除以总变化'),
        (146, 'VSUMN30', 'alpha158', '30日内成交量下跌之和除以总变化'),
        (147, 'VSUMN60', 'alpha158', '60日内成交量下跌之和除以总变化'),

        # VSUMD 因子 (ID: 148-152, 5个窗口)
        (148, 'VSUMD5', 'alpha158', '5日内成交量上涨与下跌之差除以总变化'),
        (149, 'VSUMD10', 'alpha158', '10日内成交量上涨与下跌之差除以总变化'),
        (150, 'VSUMD20', 'alpha158', '20日内成交量上涨与下跌之差除以总变化'),
        (151, 'VSUMD30', 'alpha158', '30日内成交量上涨与下跌之差除以总变化'),
        (152, 'VSUMD60', 'alpha158', '60日内成交量上涨与下跌之差除以总变化'),
    ]

    def __init__(self, params: dict = None):
        super().__init__("Alpha158", params)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有Alpha158因子，返回包含原始数据和所有因子的DataFrame"""

        # 使用字典收集所有因子列，最后一次性合并，避免DataFrame碎片化
        factors = {}

        # ==================== KBar 因子 ====================

        # 因子：收盘价与开盘价的差值除以收盘价
        # 公式：($close - $open) / $close
        factors['KMID'] = (data['close'] - data['open']) / data['close']

        # 因子：最高价与最低价的差值除以开盘价
        # 公式：($high - $low) / $open
        factors['KLEN'] = (data['high'] - data['low']) / data['open']

        # 因子：收盘价与开盘价的差值除以振幅
        # 公式：($close - $open) / ($high - $low + 1e-12)
        factors['KMID2'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-12)

        # 因子：最高价与（开盘价、收盘价较大值）的差值除以开盘价
        # 公式：($high - Greater($open, $close)) / $open
        max_oc = data[['open', 'close']].max(axis=1)
        factors['KUP'] = (data['high'] - max_oc) / data['open']

        # 因子：最高价与（开盘价、收盘价较大值）的差值除以振幅
        # 公式：($high - Greater($open, $close)) / ($high - $low + 1e-12)
        factors['KUP2'] = (data['high'] - max_oc) / (data['high'] - data['low'] + 1e-12)

        # 因子：（开盘价、收盘价较小值）与最低价的差值除以开盘价
        # 公式：(Less($open, $close) - $low) / $open
        min_oc = data[['open', 'close']].min(axis=1)
        factors['KLOW'] = (min_oc - data['low']) / data['open']

        # 因子：（开盘价、收盘价较小值）与最低价的差值除以振幅
        # 公式：(Less($open, $close) - $low) / ($high - $low + 1e-12)
        factors['KLOW2'] = (min_oc - data['low']) / (data['high'] - data['low'] + 1e-12)

        # 因子：2倍收盘价减去最高价和最低价的差值除以开盘价
        # 公式：(2*$close - $high - $low) / $open
        factors['KSFT'] = (2 * data['close'] - data['high'] - data['low']) / data['open']

        # 因子：2倍收盘价减去最高价和最低价的差值除以振幅
        # 公式：(2*$close - $high - $low) / ($high - $low + 1e-12)
        factors['KSFT2'] = (2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low'] + 1e-12)

        # ==================== Price 因子 ====================

        # 因子：开盘价除以收盘价
        # 公式：$open / $close
        factors['OPEN0'] = data['open'] / data['close']

        # 因子：最高价除以收盘价
        # 公式：$high / $close
        factors['HIGH0'] = data['high'] / data['close']

        # 因子：最低价除以收盘价
        # 公式：$low / $close
        factors['LOW0'] = data['low'] / data['close']

        # # 因子：成交量加权平均价除以收盘价
        # # 公式：VWAP / $close
        # vwap = (data['high'] + data['low'] + data['close']) / 3
        # factors['VWAP0'] = vwap / data['close']

        # # ==================== Volume 因子 ====================

        # # 因子：成交量
        # # 公式：$volume / ($volume + 1e-12)
        # factors['VOLUME0'] = data['volume'] / (data['volume'] + 1e-12)

        # ==================== ROC 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'ROC{w}'] = data['close'].shift(w) / data['close']

        # ==================== MA 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'MA{w}'] = data['close'].rolling(window=w).mean() / data['close']

        # ==================== STD 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'STD{w}'] = data['close'].rolling(window=w).std() / data['close']

        # ==================== BETA 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'BETA{w}'] = self._calculate_slope(data['close'], w) / data['close']

        # ==================== RSQR 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'RSQR{w}'] = self._calculate_rsquare(data['close'], w)

        # ==================== RESI 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'RESI{w}'] = self._calculate_residual(data['close'], w) / data['close']

        # ==================== MAX 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'MAX{w}'] = data['high'].rolling(window=w).max() / data['close']

        # ==================== MIN 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'MIN{w}'] = data['low'].rolling(window=w).min() / data['close']

        # ==================== QTLU 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'QTLU{w}'] = data['close'].rolling(window=w).quantile(0.8) / data['close']

        # ==================== QTLD 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'QTLD{w}'] = data['close'].rolling(window=w).quantile(0.2) / data['close']

        # ==================== RSV 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            high_max = data['high'].rolling(window=w).max()
            low_min = data['low'].rolling(window=w).min()
            factors[f'RSV{w}'] = (data['close'] - low_min) / (high_max - low_min + 1e-12)

        # ==================== IMAX 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'IMAX{w}'] = self._calculate_idxmax(data['high'], w) / w

        # ==================== IMIN 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'IMIN{w}'] = self._calculate_idxmin(data['low'], w) / w

        # ==================== IMXD 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            idxmax = self._calculate_idxmax(data['high'], w)
            idxmin = self._calculate_idxmin(data['low'], w)
            factors[f'IMXD{w}'] = (idxmax - idxmin) / w

        # ==================== CORR 因子 ====================

        log_volume = np.log(data['volume'] + 1)

        for w in [5, 10, 20, 30, 60]:
            factors[f'CORR{w}'] = data['close'].rolling(window=w).corr(log_volume)

        # ==================== CORD 因子 ====================

        price_change = data['close'] / data['close'].shift(1)
        volume_change = data['volume'] / data['volume'].shift(1) + 1
        log_volume_change = np.log(volume_change)

        for w in [5, 10, 20, 30, 60]:
            factors[f'CORD{w}'] = price_change.rolling(window=w).corr(log_volume_change)

        # ==================== CNTP 因子 ====================

        up_days = (data['close'] > data['close'].shift(1)).astype(float)

        for w in [5, 10, 20, 30, 60]:
            factors[f'CNTP{w}'] = up_days.rolling(window=w).mean()

        # ==================== CNTN 因子 ====================

        down_days = (data['close'] < data['close'].shift(1)).astype(float)

        for w in [5, 10, 20, 30, 60]:
            factors[f'CNTN{w}'] = down_days.rolling(window=w).mean()

        # ==================== CNTD 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'CNTD{w}'] = up_days.rolling(window=w).mean() - down_days.rolling(window=w).mean()

        # ==================== SUMP 因子 ====================

        price_change = data['close'] - data['close'].shift(1)
        gains = np.where(price_change > 0, price_change, 0)
        total_change = np.abs(price_change)

        for w in [5, 10, 20, 30, 60]:
            sum_gains = pd.Series(gains, index=data.index).rolling(window=w).sum()
            sum_total = pd.Series(total_change, index=data.index).rolling(window=w).sum()
            factors[f'SUMP{w}'] = sum_gains / (sum_total + 1e-12)

        # ==================== SUMN 因子 ====================

        losses = np.where(price_change < 0, -price_change, 0)

        for w in [5, 10, 20, 30, 60]:
            sum_losses = pd.Series(losses, index=data.index).rolling(window=w).sum()
            sum_total = pd.Series(total_change, index=data.index).rolling(window=w).sum()
            factors[f'SUMN{w}'] = sum_losses / (sum_total + 1e-12)

        # ==================== SUMD 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            sum_gains = pd.Series(gains, index=data.index).rolling(window=w).sum()
            sum_losses = pd.Series(losses, index=data.index).rolling(window=w).sum()
            sum_total = pd.Series(total_change, index=data.index).rolling(window=w).sum()
            factors[f'SUMD{w}'] = (sum_gains - sum_losses) / (sum_total + 1e-12)

        # ==================== VMA 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'VMA{w}'] = data['volume'].rolling(window=w).mean() / (data['volume'] + 1e-12)

        # ==================== VSTD 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            factors[f'VSTD{w}'] = data['volume'].rolling(window=w).std() / (data['volume'] + 1e-12)

        # ==================== WVMA 因子 ====================

        price_change_abs = np.abs(data['close'] / data['close'].shift(1) - 1)
        weighted_vol = price_change_abs * data['volume']

        for w in [5, 10, 20, 30, 60]:
            factors[f'WVMA{w}'] = weighted_vol.rolling(window=w).std() / (weighted_vol.rolling(window=w).mean() + 1e-12)

        # ==================== VSUMP 因子 ====================

        volume_change = data['volume'] - data['volume'].shift(1)
        vol_gains = np.where(volume_change > 0, volume_change, 0)
        total_vol_change = np.abs(volume_change)

        for w in [5, 10, 20, 30, 60]:
            sum_gains = pd.Series(vol_gains, index=data.index).rolling(window=w).sum()
            sum_total = pd.Series(total_vol_change, index=data.index).rolling(window=w).sum()
            factors[f'VSUMP{w}'] = sum_gains / (sum_total + 1e-12)

        # ==================== VSUMN 因子 ====================

        vol_losses = np.where(volume_change < 0, -volume_change, 0)

        for w in [5, 10, 20, 30, 60]:
            sum_losses = pd.Series(vol_losses, index=data.index).rolling(window=w).sum()
            sum_total = pd.Series(total_vol_change, index=data.index).rolling(window=w).sum()
            factors[f'VSUMN{w}'] = sum_losses / (sum_total + 1e-12)

        # ==================== VSUMD 因子 ====================

        for w in [5, 10, 20, 30, 60]:
            sum_gains = pd.Series(vol_gains, index=data.index).rolling(window=w).sum()
            sum_losses = pd.Series(vol_losses, index=data.index).rolling(window=w).sum()
            sum_total = pd.Series(total_vol_change, index=data.index).rolling(window=w).sum()
            factors[f'VSUMD{w}'] = (sum_gains - sum_losses) / (sum_total + 1e-12)

        # 将所有因子列一次性合并到原始DataFrame
        factor_df = pd.DataFrame(factors, index=data.index)
        return pd.concat([data, factor_df], axis=1)

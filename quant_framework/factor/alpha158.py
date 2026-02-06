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
    DEFINITIONS = [
        # KBar 因子
        ('KMID', 'alpha158', '收盘价与开盘价的差值除以收盘价'),
        ('KLEN', 'alpha158', '最高价与最低价的差值除以开盘价'),
        ('KMID2', 'alpha158', '收盘价与开盘价的差值除以振幅'),
        ('KUP', 'alpha158', '最高价与（开盘价、收盘价较大值）的差值除以开盘价'),
        ('KUP2', 'alpha158', '最高价与（开盘价、收盘价较大值）的差值除以振幅'),
        ('KLOW', 'alpha158', '（开盘价、收盘价较小值）与最低价的差值除以开盘价'),
        ('KLOW2', 'alpha158', '（开盘价、收盘价较小值）与最低价的差值除以振幅'),
        ('KSFT', 'alpha158', '2倍收盘价减去最高价和最低价的差值除以开盘价'),
        ('KSFT2', 'alpha158', '2倍收盘价减去最高价和最低价的差值除以振幅'),

        # Price 因子
        ('OPEN0', 'alpha158', '开盘价除以收盘价'),
        ('HIGH0', 'alpha158', '最高价除以收盘价'),
        ('LOW0', 'alpha158', '最低价除以收盘价'),

        # ROC 因子 (5个窗口)
        *[(f'ROC{w}', 'alpha158', f'{w}日收益率') for w in [5, 10, 20, 30, 60]],

        # MA 因子 (5个窗口)
        *[(f'MA{w}', 'alpha158', f'{w}日移动平均除以收盘价') for w in [5, 10, 20, 30, 60]],

        # STD 因子 (5个窗口)
        *[(f'STD{w}', 'alpha158', f'{w}日标准差除以收盘价') for w in [5, 10, 20, 30, 60]],

        # BETA 因子 (5个窗口)
        *[(f'BETA{w}', 'alpha158', f'{w}日线性回归斜率除以收盘价') for w in [5, 10, 20, 30, 60]],

        # RSQR 因子 (5个窗口)
        *[(f'RSQR{w}', 'alpha158', f'{w}日线性回归R平方') for w in [5, 10, 20, 30, 60]],

        # RESI 因子 (5个窗口)
        *[(f'RESI{w}', 'alpha158', f'{w}日线性回归残差除以收盘价') for w in [5, 10, 20, 30, 60]],

        # MAX 因子 (5个窗口)
        *[(f'MAX{w}', 'alpha158', f'{w}日最高价除以收盘价') for w in [5, 10, 20, 30, 60]],

        # MIN 因子 (5个窗口)
        *[(f'MIN{w}', 'alpha158', f'{w}日最低价除以收盘价') for w in [5, 10, 20, 30, 60]],

        # QTLU 因子 (5个窗口)
        *[(f'QTLU{w}', 'alpha158', f'{w}日80%分位数除以收盘价') for w in [5, 10, 20, 30, 60]],

        # QTLD 因子 (5个窗口)
        *[(f'QTLD{w}', 'alpha158', f'{w}日20%分位数除以收盘价') for w in [5, 10, 20, 30, 60]],

        # RSV 因子 (5个窗口)
        *[(f'RSV{w}', 'alpha158', f'{w}日RSV') for w in [5, 10, 20, 30, 60]],

        # IMAX 因子 (5个窗口)
        *[(f'IMAX{w}', 'alpha158', f'{w}日内最高价位置除以窗口长度') for w in [5, 10, 20, 30, 60]],

        # IMIN 因子 (5个窗口)
        *[(f'IMIN{w}', 'alpha158', f'{w}日内最低价位置除以窗口长度') for w in [5, 10, 20, 30, 60]],

        # IMXD 因子 (5个窗口)
        *[(f'IMXD{w}', 'alpha158', f'{w}日内最高价与最低价位置之差除以窗口长度') for w in [5, 10, 20, 30, 60]],

        # CORR 因子 (5个窗口)
        *[(f'CORR{w}', 'alpha158', f'{w}日收盘价与成交量的相关系数') for w in [5, 10, 20, 30, 60]],

        # CORD 因子 (5个窗口)
        *[(f'CORD{w}', 'alpha158', f'{w}日价格变化率与成交量变化率的相关系数') for w in [5, 10, 20, 30, 60]],

        # CNTP 因子 (5个窗口)
        *[(f'CNTP{w}', 'alpha158', f'{w}日内上涨天数占比') for w in [5, 10, 20, 30, 60]],

        # CNTN 因子 (5个窗口)
        *[(f'CNTN{w}', 'alpha158', f'{w}日内下跌天数占比') for w in [5, 10, 20, 30, 60]],

        # CNTD 因子 (5个窗口)
        *[(f'CNTD{w}', 'alpha158', f'{w}日内上涨天数占比与下跌天数占比之差') for w in [5, 10, 20, 30, 60]],

        # SUMP 因子 (5个窗口)
        *[(f'SUMP{w}', 'alpha158', f'{w}日内上涨幅度之和除以总变化幅度') for w in [5, 10, 20, 30, 60]],

        # SUMN 因子 (5个窗口)
        *[(f'SUMN{w}', 'alpha158', f'{w}日内下跌幅度之和除以总变化幅度') for w in [5, 10, 20, 30, 60]],

        # SUMD 因子 (5个窗口)
        *[(f'SUMD{w}', 'alpha158', f'{w}日内上涨与下跌幅度之差除以总变化幅度') for w in [5, 10, 20, 30, 60]],

        # VMA 因子 (5个窗口)
        *[(f'VMA{w}', 'alpha158', f'{w}日平均成交量除以当日成交量') for w in [5, 10, 20, 30, 60]],

        # VSTD 因子 (5个窗口)
        *[(f'VSTD{w}', 'alpha158', f'{w}日成交量标准差除以当日成交量') for w in [5, 10, 20, 30, 60]],

        # WVMA 因子 (5个窗口)
        *[(f'WVMA{w}', 'alpha158', f'{w}日加权成交量标准差除以加权平均成交量') for w in [5, 10, 20, 30, 60]],

        # VSUMP 因子 (5个窗口)
        *[(f'VSUMP{w}', 'alpha158', f'{w}日内成交量上涨之和除以总变化') for w in [5, 10, 20, 30, 60]],

        # VSUMN 因子 (5个窗口)
        *[(f'VSUMN{w}', 'alpha158', f'{w}日内成交量下跌之和除以总变化') for w in [5, 10, 20, 30, 60]],

        # VSUMD 因子 (5个窗口)
        *[(f'VSUMD{w}', 'alpha158', f'{w}日内成交量上涨与下跌之差除以总变化') for w in [5, 10, 20, 30, 60]],
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

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

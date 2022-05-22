'''
Stock TechnicalAnalytics by Volatility.
'''

import pandas as pd
import numpy as np
import talib
import ta

def ATR(
    pds_high,
    pds_low,
    pds_close,
    timeperiod = 14,
    fillna = None):
    '''
    AverageTrueRange.
    taライブラリによる下記コードでNaN埋めが機能せずに0埋めされてるので自作．
    df["atr"] = ta.volatility.AverageTrueRange(
        df["High"],
        df["Low"],
        df["Close"],
        window = 14,
        fillna = False
    ).average_true_range()

    - Refs
        - https://jp.tradingview.com/scripts/averagetruerange/?solution=43000501823
    '''
    min_periods = timeperiod
    list_tr = []
    if fillna:
        min_periods = 1
        list_tr = [fillna for _ in range(timeperiod - 1)]
    else:
        list_tr = [np.NaN for _ in range(timeperiod - 1)]

    for i in range(timeperiod - 2, len(pds_high) - 1):
        list_tr.append(
            max(
                pds_high[i + 1] - pds_low[i + 1],
                abs(pds_high[i + 1] - pds_close[i]),
                abs(pds_low[i + 1] - pds_close[i])
            )
        )
    # pds_atr = pd.Series(list_tr).ewm(span = timeperiod).mean()
    pds_tr = pd.Series(list_tr)
    pds_sma = pds_tr.rolling(timeperiod, min_periods = min_periods).mean()[:timeperiod]
    pds_atr = pd.concat([pds_sma, pds_tr[timeperiod:]]).ewm(span = timeperiod, adjust = False).mean()
    return pds_atr

def CHOPPINESS(
    pds_high,
    pds_low,
    pds_close,
    timeperiod = 14,
    fillna = None):
    '''
    Choppiness Index.
    - Refs
        - https://jp.tradingview.com/scripts/choppinessindex/?solution=43000501980
        - https://bitcoin-talk.info/choppiness-index/
        - https://note.nkmk.me/python-pandas-rolling/
    - TODO
        - fillnaしたい時ってローリング長満たしてなくてもあるもので計算するもの？それとも0とかの固定値にnaを置換するもの？どっちが正しい？
    '''
    min_periods = timeperiod
    if fillna:
        min_periods = 1
    pds_max_high = pds_high \
        .rolling(timeperiod, min_periods = min_periods) \
        .max()
    pds_max_low = pds_low \
        .rolling(timeperiod, min_periods = min_periods) \
        .min()
    pds_sum_atr = ta.volatility.AverageTrueRange(
        pds_high,
        pds_low,
        pds_close,
        window = timeperiod,
        fillna = fillna).average_true_range()
    print(pds_sum_atr)
    # print(list(pds_sum_atr))
    pds_sum_atr = ATR(
        pds_high,
        pds_low,
        pds_close,
        timeperiod = timeperiod,
        fillna = None)
    print(pds_sum_atr)
    # print(list(pds_sum_atr))
    print(talib.ATR(
        pds_high,
        pds_low,
        pds_close,
        timeperiod = timeperiod
    ))
    # return 
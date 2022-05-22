'''
- Refs
    - https://investpy.readthedocs.io/_api/technical.html
    - https://oeconomicus.jp/2021/06/python-infestpy/
    - https://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html
    - https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.VortexIndicator
    - https://gist.github.com/imtaehyun/8a6223142e07eaf1ef2215de10ca7a5d#file-technical-analysis-indicators-without-talib-code-py-L155

TODO: 本来はfeatures_processへ関数化すべき．
'''

import inspect
import os
import sys
PYPATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOTPATH = PYPATH + "/./.." # `main`をrootにする
sys.path.append(ROOTPATH)

from argparse import ArgumentParser
import pandas as pd
import investpy
import talib
import ta

from modules_ml.data_process.with_pandas.stock.ta_volatility import (
    CHOPPINESS
)


def main(symbol):
    dir_downloads = "{}/inputs/downloads".format(ROOTPATH)
    dir_data = "{}/inputs/data".format(ROOTPATH)
    os.makedirs(dir_data, exist_ok = True)
    # ダウンロード済みCSVを読み込み
    df = pd.read_csv(
        "{0}/stock_of_{1}_daily_from_investingcom.csv".format(
            dir_downloads,
            symbol
        ),
        header = 0,
        encoding = "utf8"
    )

    # ボリンジャーバンドの集計結果のカラム追加
    ## https://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/bb
    df["bb_2_upper"], df["bb_2_mid"], df["bb_2_lower"] = talib.BBANDS(
        df["Close"],
        # ひとまずinvesting.comでの設定値に合わせる
        timeperiod = 20,
        nbdevup = 2,
        nbdevdn = 2,
        matype = 0
    )
    # print(df)
    print(df.shape)

    # MACDの集計結果のカラム追加
    ## https://www.metatrader5.com/ja/terminal/help/indicators/oscillators/macd
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
        df["Close"],
        # ひとまずinvesting.comでの設定値に合わせる
        fastperiod = 12,
        slowperiod = 26,
        signalperiod = 9
    )
    # print(df)
    print(df.shape)

    # RSIの集計結果のカラム追加
    df["rsi"] = talib.RSI(
        df["Close"],
        # ひとまずinvesting.comでの設定値に合わせる
        timeperiod = 14
    )
    # print(df)
    print(df.shape)

    # VortexIndicatorの集計結果のカラム追加
    df["vortex_pos"] = ta.trend.vortex_indicator_pos(
        df["High"],
        df["Low"],
        df["Close"],
        # ひとまずinvesting.comでの設定値に合わせる
        window = 14,
        fillna = False
    )
    df["vortex_neg"] = ta.trend.vortex_indicator_neg(
        df["High"],
        df["Low"],
        df["Close"],
        # ひとまずinvesting.comでの設定値に合わせる
        window = 14,
        fillna = False
    )
    # print(df)
    print(df.shape)

    # AwesomeOscillatorの集計結果のカラム追加
    ## https://www.metatrader5.com/ja/terminal/help/indicators/bw_indicators/awesome
    df["awesome"] = ta.momentum.AwesomeOscillatorIndicator(
        df["High"],
        df["Low"],
        # ひとまず一般的な理論に従う
        window1 = 5,
        window2 = 34,
        fillna = False
    ).awesome_oscillator()
    # print(df)
    print(df.shape)

    # (Elder's)ForceIndexの集計結果のカラム追加
    ## https://www.metatrader5.com/ja/terminal/help/indicators/oscillators/fi
    df["force"] = ta.volume.ForceIndexIndicator(
        df["Close"],
        df["Volume"],
        # ひとまずinvesting.comでの設定値に合わせる
        window = 13,
        fillna = False
    ).force_index()
    # print(df)
    print(df.shape)

    # ChoppinessIndexの集計結果のカラム追加
    CHOPPINESS(
        df["High"],
        df["Low"],
        df["Close"],
        timeperiod = 14,
        fillna = None # 0.0
    )
    # print(CHOPPINESS(
    #     df["High"],
    #     df["Low"],
    #     df["Close"],
    #     timeperiod = 14,
    #     fillna = None # 0.0
    # ))


    # CSV出力
    df.to_csv(
        "{0}/stock_of_{1}_daily_from_investingcom.csv".format(
            dir_data,
            symbol
        ),
        header = True,
        index = False,
        mode = "w",
        sep = ",",
        encoding = "utf8"
    )



if __name__ == "__main__":
    parser = parser = ArgumentParser()
    parser.add_argument(
        "-s", "--symbol",
        help = "データ加工するシンボル",
        type = str
    )
    cli_args = parser.parse_args()
    main(cli_args.symbol)
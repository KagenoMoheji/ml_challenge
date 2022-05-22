"""
- Refs
    - https://investpy.readthedocs.io/_api/technical.html
    - https://oeconomicus.jp/2021/06/python-infestpy/
"""

import inspect
import os
import sys
PYPATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOTPATH = PYPATH + "/./.." # `main`をrootにする
sys.path.append(ROOTPATH)

from argparse import ArgumentParser
import pandas as pd
import investpy


def main(symbol):
    dir_downloads = "{}/inputs/downloads".format(ROOTPATH)
    os.makedirs(dir_downloads, exist_ok = True)
    # 株価データの取得
    df = investpy.get_stock_historical_data(
        stock = symbol,
        country = "united states",
        from_date = "01/01/2015",
        to_date = "28/02/2022")
    # print(type(df))
    # print(df)

    # インデックスに日付があるのでカラムに移す
    ## df["Date"] = df.index # df.reset_index()がインデックスをカラムにしてくれるので不要
    df = df.reset_index(drop = False)
    # print(df)
    print(df.shape)

    # CSV出力
    df.to_csv(
        "{0}/stock_of_{1}_daily_from_investingcom.csv".format(
            dir_downloads,
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
        help = "ダウンロードしたいシンボル",
        type = str
    )
    cli_args = parser.parse_args()
    main(cli_args.symbol)

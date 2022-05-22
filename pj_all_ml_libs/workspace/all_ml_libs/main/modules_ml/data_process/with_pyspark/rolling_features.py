'''
- Refs
    - [第264話｜時系列データの5種類の特徴量（説明変数）](https://www.salesanalytics.co.jp/column/no00264/)
'''
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, DoubleType, BooleanType,
    DateType, TimeStampType,
    ArrayType
)
from pyspark.sql.window import Window
import pyspark.sql.functions as sparkF
import pandas as pd

from data_process.utils.with_numpy.sliding_window import np_sliding_window_view


def convert_to_series_of_rolling_features(
    spdf_data,
    colnames_feature,
    colname_seq,
    in_seq_len,
    out_seq_len,
    out_mask = None,
    cnt_out_mask = 1,
    colnames_out_suppls = None,
    spdf_data_replace_out = None,
    cnt_data_replace_out = 1):
    '''
    pysparkデータフレームからnumpyのローリング特徴量群を生成して返す．
    - Args
        - spdf_data:pyspark.sql.DataFrame: 
        - colnames_feature:list[str]: 特徴量とするカラム名リスト
            - 目的変数を右端としたい場合は，事前に下記のようなコードでカラム順をソートしておく必要あり．この関数ではそのカラム順でSELECTする仕様になっている．
            ```
            list_colname_sort_index = []
            for i, colname in enumerate(spdf_data.columns):
                # 特徴量として使わないカラム名を指定
                if colname in [
                    "date",
                    "day",
                    "weather"
                ]:
                    continue

                # 右に来てほしいほど負に大きい値を指定
                ## 特に目的変数(予測の出力)とするカラムは右端にすべし
                if colname == "max_temp": # 最高気温が目的変数の場合
                    colname_index = (-100, colname)
                # elif colname == "hoge": # 右側に寄せたい説明変数がある場合に指定
                #     colname_index = (-1, colname)
                else:
                    colname_index = (i, colname)
                list_colname_sort_index.append(colname_index)
            colnames_feature_sorted = [
                c[i]
                for i in sorted(list_colname_sort_index, reverse = True)
            ]
            ```
        - colname_seq:str: ローリング特徴量を作成する上で基準とする連続値のカラム名．基本的に日付カラム．
            - ただし，ローリング特徴量は連続値の中の「指定の個数」で作っており，日付の範囲指定によるグループ化をしているわけではない．
        - in_seq_len:int: 時系列予測でのローリング特徴量における，入力部分の個数．単位は基本的に日数．
        - out_seq_len:int: 時系列予測でのローリング特徴量における，出力部分の個数．単位は基本的に日数．
        - out_mask:float: ローリング特徴量における出力部分をマスクするので，その値．
            - マスキングする場合は戻り値`np_masked`にマスキングされた値のnumpyを返す．
            - Noneを渡した場合はマスキングなし．
            - `float("-inf")`とかよくありそう．
        - cnt_out_mask:int: ローリング特徴量における出力部分，つまり後ろ側からいくつ分マスクするか．
            - 出力部分を含め直近のデータを用意できない場合で用いる．
        - colnames_out_suppls:list[str]: ローリング特徴量の出力部分の補足情報として戻り値`np_out_suppls`を得たい場合に指定するカラム名リスト．
            - 不要な場合はNoneか空リストを渡す．
        - spdf_data_replace_out:pyspark.sql.DataFrame: ローリング特徴量の後方部分(基本的に出力部分)をリプレースするのに用いるDataFrame.
            - `spdf_data`と同じカラム・テーブルサイズであるべし(前処理も同じ過程を通すべし)．
            - 学習時の検証データを用いた予測において，ローリング特徴量の最後数個分の実測データを予報データに置き換えたい場合とかで用いる．
        - cnt_data_replace_out:int: `spdf_data_replace_out`を用いてローリング特徴量の後方リプレースする後方いくつ分か．
    - Returns
        - np_series_of_rolling_features:numpy.array: `in_seq_len + out_seq_len`個のローリング特徴量群の配列
        - np_masked:numpy.array: マスキングされるローリング特徴量の出力部分の配列
        - np_out_suppls:numpy.array: ローリング特徴量の出力部分の補足情報．
            - `colnames_out_suppls`がNoneか空リストの場合はNoneを返す．
            - `np_series_of_rolling_features`は`colnames_feature`に基づき特徴量のみ抽出するので日付などの情報が消えてしまう．
            - そこで日付などの情報を`np_series_of_rolling_features`・`np_masked`のインデックスに合わせてもたせるnumpy配列として別途返す形で実装した．
            - `np_out_suppls`の各配列における値の並び順は`colnames_out_suppls`の並び順に則る．
    '''
    rolling_len = in_seq_len + out_seq_len
    np_series_of_rolling_features = None
    np_masked = None
    np_out_suppls = None
    for i, 






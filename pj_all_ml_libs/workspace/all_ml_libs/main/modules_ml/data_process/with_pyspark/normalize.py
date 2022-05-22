import uuid
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, DoubleType, BooleanType,
    DateType, TimeStampType,
    ArrayType
)
from pyspark.sql.window import Window
import pyspark.sql.functions as sparkF
import pandas as pd


def spdf_normalize(
    spdf,
    colname_target,
    colname_result,
    norm_min = None,
    norm_max = None):
    '''
    - Args
        - spdf:pyspark.sql.DataFrame: 集計対象のpysparkデータフレーム
        - colname_target:str: 正規化する数値型のカラム名
        - colname_result:str: 集計結果を格納するカラム名
        - norm_min:int|float: 正規化の最小値．Noneの場合はspdfから導出する．
            - 学習時と同じ値を検証・予測時で用いて前処理を再現するために用いる．
        - norm_max:int|float: 正規化の最大値．Noneの場合はspdfから導出する．
            - 学習時と同じ値を検証・予測時で用いて前処理を再現するために用いる．
    - TODO
        - NaNやNaTなどの欠損値を考慮していない．ひとまずspdfにデータ埋まっている想定．
    '''
    tmp_id = str(uuid.uuid4())[:8]
    colname_min = "{target}_min_{id}".format(
        target = colname_target,
        id = tmp_id)
    colname_max = "{target}_max_{id}".format(
        target = colname_target,
        id = tmp_id)
    if norm_min is None:
        norm_min = spdf.agg(sparkF.min(sparkF.col(colname_target)).alias(colname_min)).collect()[0][colname_min]
        # norm_min = spdf.select(sparkF.min(sparkF.col(colname_target)).alias(colname_min)).collect()[0][colname_min]
    if norm_max is None:
        norm_max = spdf.agg(sparkF.max(sparkF.col(colname_target)).alias(colname_max)).collect()[0][colname_max]
        # norm_max = spdf.select(sparkF.max(sparkF.col(colname_target)).alias(colname_max)).collect()[0][colname_max]
    spdf = spdf \
        .withColumn(
            colname_min,
            sparkF.lit(norm_min)) \
        .withColumn(
            colname_max,
            sparkF.lit(norm_max)) \
        .withColumn(
            colname_result,
            (sparkF.col(colname_target) - sparkF.col(colname_min)) / (sparkF.col(colname_max) - sparkF.col(colname_min))) \
        .drop(colname_min, colname_max)
    return spdf, norm_min, norm_max


def spdf_normalize_in_partition(
    spdf,
    colname_target,
    colname_result,
    colnames_partitionby):
    '''
    - Args
        - spdf:pyspark.sql.DataFrame: 集計対象のpysparkデータフレーム
        - colname_target:str: 正規化する数値型のカラム名
        - colname_result:str: 集計結果を格納するカラム名
        - colnames_partitionby:list[str]: 正規化に必要な最大値・最小値を得るためにpartitionbyするカラム名のリスト
    - TODO
        - NaNやNaTなどの欠損値を考慮していない．ひとまずspdfにデータ埋まっている想定．
        - パーティションごとの最大値・最小値を戻り値・引数で受け取り，検証・予測において再現できるようにする必要あるかも…
    '''
    tmp_id = str(uuid.uuid4())[:8]
    colname_min = "{target}_min_{id}".format(
        target = colname_target,
        id = tmp_id)
    colname_max = "{target}_max_{id}".format(
        target = colname_target,
        id = tmp_id)
    w = Window.partitionBy(colnames_partitionby)
    spdf = spdf \
        .withColumn(
            colname_min,
            sparkF.min(sparkF.col(colname_target)).over(w)) \
        .withColumn(
            colname_max,
            sparkF.max(sparkF.col(colname_target)).over(w)) \
        .withColumn(
            colname_result,
            (sparkF.col(colname_target) - sparkF.col(colname_min)) / (sparkF.col(colname_max) - sparkF.col(colname_min))) \
        .drop(colname_min, colname_max)
    return spdf



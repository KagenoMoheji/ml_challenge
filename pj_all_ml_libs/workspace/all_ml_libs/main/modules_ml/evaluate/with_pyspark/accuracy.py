from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, DoubleType, BooleanType,
    DateType, TimeStampType,
    ArrayType
)
from pyspark.sql.window import Window
import pyspark.sql.functions as sparkF
import pandas as pd

def accuracy_in_error_range(
    spdf,
    colname_true,
    colname_pred,
    colnames_groupby,
    error_rate,
    round_error_range = False):
    '''
    誤差範囲内を正解とした正解率を導出．
    - Args
        - spdf:pyspark.sql.DataFrame: 
        - colname_true:str: 正解ラベルのカラム名
        - colname_pred:str: 予測結果のカラム名
        - colnames_groupby:list[str]: 個別のaccuracyを求めるためのgroupbyのカラム名リスト
        - error_rate:float: 前後誤差率．前後10%以内とする場合は0.1を指定．
        - round_error_range:bool: 誤差範囲の上限下限を四捨五入するか
    - Returns
        - spdf:pyspark.sql.DataFrame: 引数`spdf`に正解範囲の2カラム・正解かの真偽値カラムを追加したDataFrame
        - spdf_acc_groupby:pyspark.sql.DataFrame: 引数`colname_groupby`を用いて個別のaccuracyを求めた結果のDataFrame
        - acc_all:float: `spdf`全データにおけるaccuracy
    - TODO
        - 極端，正解値2の時誤差範囲が「2以上2以下」になってピンポイントで正解しなくてはならなくなるのでは？
            - [A]極端に小さい数字が発生する非営業日を含めないので，考慮しない方針らしい．
    '''
    colname_range_min = "range_min_of_{}".format(colname_true)
    colname_range_max = "range_max_of_{}".format(colname_true)
    spdf = spdf \
        .withColumn(
            colname_range_min,
            sparkF.col(colname_true) - sparkF.col(colname_true) * error_rate if not round_error_range
            else sparkF.round(sparkF.col(colname_true) - sparkF.col(colname_true) * error_rate)
        ) \
        .withColumn(
            colname_range_max,
            sparkF.col(colname_true) + sparkF.col(colname_true) * error_rate if not round_error_range
            else sparkF.round(sparkF.col(colname_true) + sparkF.col(colname_true) * error_rate)
        ) \
        .withColumn(
            "is_correct",
            sparkF.col(colname_pred).between(sparkF.col(colname_range_min), sparkF.col(colname_range_max))
        )
        # .withColumn(
        #     "is_correct",
        #     (sparkF.col(colname_range_min) <= sparkF.col(colname_pred))
        #     & (sparkF.col(colname_pred) <= sparkF.col(colname_range_max))
        # )
    spdf_acc_groupby = spdf \
        .groupBy(colnames_groupby) \
        .agg(
            (
                sparkF.sum(
                    sparkF \
                        .when(sparkF.col("is_correct"), 1) \
                        .otherwise(0)
                ) / sparkF.count(sparkF.lit(1))
            ).alias("accuracy")
        )
    acc_all = spdf \
        .where(sparkF.col("is_correct")) \
        .count() / spdf.count()
    return spdf, spdf_acc_groupby, acc_all




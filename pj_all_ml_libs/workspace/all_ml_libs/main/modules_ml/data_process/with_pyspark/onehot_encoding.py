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

def spdf_agg_onehot_encode(map_cat_onehot):
    '''
    渡されるカテゴリ変数のカラム(pd.Series)のデータをmap_cat_onehotの対応表が網羅している前提で，OneHotEncoding結果のリストを格納する．
    - Args
        - map_cat_onehot:list: `[{"cat_name": ~, "cat_id": ~, "onehot": [~]}, ...]`形式のカテゴリ変数とOneHotEncoding結果の対応表．
    - ここでエラーが出る場合，考えられること
        - map_cat_onehotの対応表がカテゴリ変数のカラムのデータを網羅していない
        - map_cat_onehotの対応表に登録しているカテゴリ値と一致しないカテゴリ値がデータに存在する
    - TODO
        - map_cat_onehot["onehot"]とmap_cat_onehot["cat_id"]の一意性の保証してない
    '''
    @sparkF.pandas_udf(ArrayType(IntegerType()))
    def udf_onehot_encode(pds: pd.Series) -> pd.Series:
        return pds.apply(lambda row: [cat for cat in map_cat_onehot if cat["cat_name"] == row][0]["onehot"])
    return udf_onehot_encode

def spdf_onehot_encode(
    spdf,
    colname_cat,
    flatten_onehot_into_cols = False,
    replace_cat2onehot = False,
    map_cat_onehot = None):
    '''
    - Args
        - spdf:pyspark.sql.DataFrame: 
        - colname_cat:str: OneHotEncoding元のカテゴリ変数のカラム名
        - flatten_onehot_into_cols:bool: OneHotEncoding結果を複数カラムに展開するか．
            - デフォルトはFalseであり1つのArray型カラムにOneHotEncoding結果が格納される．
        - replace_cat2onehot:bool: OneHotEncoding元のカラムを，OneHotEncoding結果に置き換えるか．
            - `flatten_onehot_into_cols = True`の場合は元のカラムが削除される．
        - map_cat_onehot:list: `[{"cat_name": ~, "cat_id": ~, "onehot": [~]}, ...]`形式のカテゴリ変数とOneHotEncoding結果の対応表．渡された場合はこの対応表に則って処理する．
            - 学習時と同じ値を検証・予測時で用いて前処理を再現するために用いる．
    '''
    if map_cat_onehot is None:
        map_cat_onehot = []
        # cat_names = [row[colname_cat] for row in spdf.select(sparkF.col(colname_cat)).distinct().rdd.flatMap(lambda x: x).collect()]
        cat_names = [row[colname_cat] for row in spdf.select(sparkF.col(colname_cat)).distinct().collect()]
        for i, cat_name in enumerate(cat_names):
            map_cat_onehot.append({
                "cat_name": cat_name,
                "cat_id": str(uuid.uuid4())[:8],
                "onehot": [1 if j == i else 0 for j in reversed(range(len(cat_names)))]
            })
    else:
        if any([not cat.get("cat_id") for cat in map_cat_onehot]):
            # 1つでも空文字またはcat_idキーなしのカテゴリ変数対応データがあったら，全てのcatr_idをuuidで刷新
            for cat in map_cat_onehot:
                cat["cat_id"] = str(uuid.uuid4())[:8]
    # print(map_cat_onehot)
    colname_onehot = "{}_onehot".format(colname_cat)
    if replace_cat2onehot:
        colname_onehot = colname_cat
    for cat in map_cat_onehot:
        spdf_result = spdf \
            .withColumn(
                colname_onehot,
                spdf_agg_onehot_encode(map_cat_onehot)(sparkF.col(colname_cat)))
    if flatten_onehot_into_cols:
        for i, cat in enumerate(map_cat_onehot):
            spdf_result = spdf_result \
                .withColumn(
                    "{}_{}".format(colname_cat, cat["cat_id"]),
                    sparkF.lit(sparkF.col(colname_onehot)[i]))
        spdf_result = spdf_result.drop(sparkF.col(colname_onehot))
    # spdf_result.show() # display(spdf_result)
    return spdf_result, map_cat_onehot


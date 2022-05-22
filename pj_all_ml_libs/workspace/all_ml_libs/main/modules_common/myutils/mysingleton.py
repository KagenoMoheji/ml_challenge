class Singleton(object):
    '''
    シングルトンクラス。
    [継承するサブクラスにおける実装ルール]
    ・初回インスタンス生成時のみ実行する処理(フィールドに格納するなど)は、Singletonクラスのサブクラスにおいてprivate関数"_init_instance()"にオーバーライド(@staticmethod付けない！)する形で行う。
    ・"_init_instance()"に呼び出される初回インスタンス生成時のみに呼び出されるその他関数(たぶん@staticmethod付けない方がいい)の関数名は、"_"で始まるものとする。
    ・メソッドは基本"@staticmethod"を付けて静的にする。
    ・private関数の関数名は"__”で始まるものとする。
    '''
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
#             print("Singleton初回作成")
            # インスタンス生成されていなかったらここで生成。したがってインスタンス生成済みなら再生成されない。
            cls._instance = super(Singleton, cls).__new__(cls)
            # インスタンス生成時のみに実行する関数を実行。この関数では例えばフィールド(インスタンス変数)に値を格納するとか。
            cls._instance._init_instance(*args, **kwargs)
#         else:
#             print("Singleton作成済みです")
        return cls._instance

    def _init_instance(*args, **kwargs):
        pass


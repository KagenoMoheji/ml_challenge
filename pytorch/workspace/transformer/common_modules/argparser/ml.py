from argparse import ArgumentParser

def get_ml_parser(description):
    parser = ArgumentParser(description = description)
    parser.add_argument(
        "-p", "--pred",
        help = "学習済みモデルを読み込んで予測するモードか．オプション無しの場合は学習する",
        action = "store_true"
    )
    parser.add_argument(
        "-m", "--model",
        help = "学習して保存する，または読み込んで予測するモデル名．拡張子なし．",
        required = True,
        type = str
    )
    return parser

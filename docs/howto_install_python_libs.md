# Pythonライブラリのインストール
### `TA-Lib`
- ~~Windowsで普通にpip installできない．~~
    - [公式](https://github.com/mrjbq7/ta-lib)ではバージョン古そうなWindows用ビルダーをダウンロードしろとのことだが，[非公式のライブラリダウンローダ](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)からWindows版でPythonバージョンを合わせたファイルをダウンロードして下記コマンドでインストールできそう．
        ```
        pipenv run python -m pip install --no-index --find-links .\libs\ TA-Lib
        ```
    - [20220522]なんかやり方変わったっぽい．
        - **というかLinuxでやることにしたので今後Windows版は世話にならんかも．**


### `torch`
- [インストーラ配布サイト](https://download.pytorch.org/whl/cpu/torch_stable.html)からOS・Pythonバージョン・CPU/GPU版を合わせてダウンロードして，インストールする．
- CPU版Windows
    - Python3.9なら下記コマンドでインストール
        ```
        pip install https://download.pytorch.org/whl/cpu/torch-1.10.2%2Bcpu-cp39-cp39-win_amd64.whl
        ```
- CPU版Linux
    - Docker(Ubuntu)でPython3.9なら下記コマンドでインストール
        ```
        pip install https://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp39-cp39-linux_x86_64.whl
        ```

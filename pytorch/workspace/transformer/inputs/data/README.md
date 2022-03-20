- `weather_tokyo_daily_from_jma.csv`
    - 東京の天気を日次・1年分
    - 取得方法
        1. [過去の気象データ・ダウンロード | 気象庁](https://www.data.jma.go.jp/risk/obsdl/index.php#)からダウンロード
        2. ファイル名変更
- `nikkei_stock_average_daily_jp.csv`
    - 日経平均株価を日次・2年分？
    - 取得方法
        1. [ダウンロードセンター | 日経平均プロフィル](https://indexes.nikkei.co.jp/nkave/index?type=download)からダウンロード
        2. ファイル名変更
- `stock_of_{シンボル}_daily_from_investingcom.csv`
    - {シンボル}株の株価を日次・数年分
    - 取得方法
        1. `investing_com.py`でAPI取得
        2. `TA-Lib`や`ta`のライブラリを用いてテクニカル指標を追加
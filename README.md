## ディレクトリルール
```
├ ml_challenge/
    ├ .gitignore
    ├ README.md
    ├ docs/ [ドキュメント]
    ├ pj_{PJ名}/
        ├ README.md
        ├ docker-compose.yml
        ├ .env
        ├ .dockerignore
        ├ dcup.bash
        ├ dexec.bash
        ├ docker/
            ├ saved_images/ [ライブラリインストール済みまでならこっちで十分かも]
            ├ exported_containers/ [あまり使わなさそう]
            ├ {Dockerサービス名}/
            ├ dev/
                ├ Dockerfile
                ├ tools/
                    ├ python/
                        ├ install_requirements.bash
                        ├ user_requirements.txt
                        ├ libs/ [user_requirements.txtでインストールさせるwhlやtargz]
        ├ workspace/
            ├ {手法・アプローチ名}/
                ├ README.md
                ├ {使用する主要ライブラリ・環境名}/
                    ├ README.md
                    ├ Pipfile
                    ├ Pipfile.lock
                    ├ (requirements.txt)
                    ├ (pipdeptree.txt)
                    ├ libs/ [Pipfileやrequirements.txtでインストールさせるwhlやtargz]
                    ├ main/
                        ├ main_{詳細の手法・アプローチ名}.py
                        ├ proc_{工程名}/ [prc=process,procedure]
                            ├ main_{詳細の手法・アプローチ名}.py
                        ├ modules_common/ [機械学習関連以外の共通処理]
                        ├ modules_ml/ [機械学習関連の共通処理]
                            ├ models/ [モデル定義]
                            ├ (layers/ [NN層定義．modelsのコードで使う．])
                            ├ features_process/ [入力データ・前処理パラメータ・学習モードフラグを引数とし，複数の前処理を通して，X・y・補足情報(np_suppl(ement)s)・前処理パラメータを戻り値とする関数]
                            ├ data_process/ [前処理]
                                ├ with_{ライブラリ名}/
                                    ├ ({カテゴリラベル}/)
                                ├ with_pandas/
                                    ├ onehot_encoding.py
                                ├ with_numpy/
                                ├ with_pyspark/
                            ├ evaluate/
                        ├ outputs/
                            ├ models/ [学習済みモデルのファイル出力先]
                                ├ {モデル名}/
                                    ├ model.{model|h5|bin|pth|onnx}
                                    ├ model_params.json [モデル定義のコンストラクタ引数・前処理パラメータ・入力カラム順など]
                        ├ inputs/
                            ├ downloads/ [APIなどで取ってきた生データ]
                            ├ data/ [特徴量生成経過や特徴量として完成状態のデータ]
                    ├ mytest/
```
- `{使用する主要ライブラリ・環境名}/`について，サーバ等への商用リリースが無い前提ならば`main/{使用する主要ライブラリ・環境名}/`にして，Pipfileに複数の同種ライブラリ(tensorflowやtorch等)をインストールして開発しても良いかもしれない．

## Dockerコンテナの作り方
1. 下記ファイルを作成
    - `.env`
    - `docker-compose.yml`
    - `Dockerfile`
2. 下記コマンドでイメージビルド＆コンテナ起動
    ```
    $ cd ml_challenge/{pj_PJ名}
    $ sh dcup.bash
    ```
3. 下記コマンドで個人ユーザでコンテナに入る
    ```
    $ sh dexec.bash -u current -c {コンテナ名}
    ```
4. Python仮想環境のPipfileが未作成の場合は下記コマンドで手動インストールして環境整備
    ```
    # cd {手法・アプローチ名}/{使用する主要ライブラリ・環境名}
    # pipenv --python {Pythonバージョン}
    # pipenv install {ライブラリ名}
    ```
5. 環境整備が終わったら下記コマンドでPython仮想環境を作り直して依存ライブラリを出力
    ```
    # pipenv --rm
    # rm Pipfile.lock
    # pipenv install [=> ここでバージョン衝突とかでエラー出た場合は，ライブラリを個別インストールし直す方針にする]
    # pipenv run python -m pip freeze > requirements.txt
    # pipenv run python -m pipdeptree -fl > requirements_pipdeptree.txt
    # pipenv run python -m pipdeptree > pipdeptree.txt
    ```
6. 以下，Dockerコンテナに対しいくつのPythonプロジェクトを持つかでコンテナ・仮想環境の用意の仕方が変わる．
    - DockerコンテナにおいてPythonプロジェクトを複数実装し，各プロジェクトにPython仮想環境を用意する方針の場合．  
    なおPython仮想環境が無い時の対応．
        1. Pipfile.lockがあることが最良だが，無い場合は下記コマンドで出力したrequirements.txtを自動でPipfileに読み込んでもらう．
            ```
            # rm Pipfile
            # pipenv --python 3.9
            ```
        2. 下記コマンドで依存ライブラリをインストール
            ```
            # pipenv install
            ```
    - DockerコンテナにおけるPythonプロジェクトが単一で，Dockerコンテナに直にライブラリインストールする場合．
        1. 出力したrequirements.txtの内容を`docker/dev/tools/python/user_requirements.txt`にコピペ
        2. 下記コマンドでDockerコンテナを作り直して，Python仮想環境への依存ライブラリのインストールまで完了することを確認する
            ```
            # pipenv --rm
            $ docker-compose down
            $ docker rmi {イメージ名}
            $ docker volume prune
            $ docker network prune
            $ sh dcup.bash
            ```
            - `sh dcup.bash(=docker-compose up -d)`が始まらないなと思ったら下記の可能性あり．
                - `.venv`による大量のファイルの転送が時間かかっている
                    - `.venv`は作り直せばいいので，削除してしまえ
                - [docker-compose で一向にビルドがはじまらない、もしくは起動しない。はたまた忘れたころに起動する。](https://qiita.com/KEINOS/items/42aae92d00675c8b0b78)
        3.  出来上がったコンテナのイメージを保存し，それに基づくコンテナ起動ができるか確認する
            ```
            $ docker save {イメージ名} -o ./docker/saved_images/{イメージ名}_{日付}.tar
            $ sh dexec.bash -u current -c {コンテナ名}
            # pipenv --rm
            $ docker-compose down
            $ docker rmi {イメージ名}
            $ docker volume prune
            $ docker network prune
            $ docker load -i ./docker/saved_images/{イメージ名}_{日付}.tar
            $ sh dcup.bash
            ```
            - コンテナのファイルシステムの保存の場合は下記
                ```
                $ docker export {コンテナ名} > ./docker/exported_containers/{コンテナ名}_{日付}.tar
                ```

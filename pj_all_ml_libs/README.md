# 機械学習ライブラリ総集
使いそうな機械学習関連ライブラリを全部インストールしたDockerコンテナ作ろうぜ！！
たぶんめっちゃデカく・重くなる！！

## 環境
一旦CPU環境を想定．


## ライブラリ一覧
```
# [20220521]インストールコマンド
pipenv install \
    numpy \
    scipy \
    jax[cpu] \
    jaxlib \
    pandas \
    polars \
    modin \
    vaex \
    pandarallel \
    matplotlib \
    japanize-matplotlib \
    seaborn \
    plotly \
    streamlit \
    dash \
    dask[complete] \
    pymssql \
    psycopg2-binary \
    tensorflow \
    keras \
    https://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp39-cp39-linux_x86_64.whl \
    https://download.pytorch.org/whl/cpu/torchvision-0.12.0%2Bcpu-cp39-cp39-linux_x86_64.whl \
    https://download.pytorch.org/whl/cpu/torchaudio-0.11.0%2Bcpu-cp39-cp39-linux_x86_64.whl \
    https://download.pytorch.org/whl/torchtext-0.12.0-cp39-cp39-linux_x86_64.whl \
    https://data.pyg.org/whl/torch-1.11.0%2Bcpu/torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl \
    https://data.pyg.org/whl/torch-1.11.0%2Bcpu/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl \
    https://data.pyg.org/whl/torch-1.11.0%2Bcpu/torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl \
    https://data.pyg.org/whl/torch-1.11.0%2Bcpu/torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl \
    torch-geometric \
    mxnet \
    https://data.dgl.ai/wheels/dgl-0.8.1-cp39-cp39-manylinux1_x86_64.whl \
    dglgo \
    scikit-learn \
    lightgbm \
    pytorch-lightning \
    transformers \
    gensim \
    janome \
    fasttext \
    spacy \
    ginza \
    nlplot \
    opencv-python \
    Pillow \
    mediapipe \
    pystan \
    deap \
    python-Levenshtein \
    pipdeptree
```
### 共通
##### 配列・数値計算
- numpy
- scipy
    - https://scipy.org/install/
- ~~theano~~
    - 開発中止

##### データフレーム・表計算
- pandas
- polars
    - https://pola-rs.github.io/polars-book/user-guide/quickstart/intro.html
- modin
    - https://modin.readthedocs.io/en/stable/#installation-and-choosing-your-compute-engine
- vaex
    - https://vaex.io/docs/installing.html
- pandarallel
    - https://github.com/nalepae/pandarallel#installation
- ~~cudf~~
    - GPUがいるっぽいし，pipでインストールできなさそうなのでインストールしない
- ~~pyspark~~
    - Sparkのインストールが必要なのでインストールしない

##### グラフ描写
- matplotlib
- japanize-matplotlib
- seaborn
    - https://seaborn.pydata.org/installing.html
- plotly
    - https://plotly.com/python/getting-started/

##### Webビルダー
- streamlit
    - https://docs.streamlit.io/library/get-started/installation#install-streamlit-on-macoslinux
- dash
    - https://dash.plotly.com/installation

##### 並列処理
- dask
    - 配列・数値計算とデータフレーム・表計算の両方対応
    - https://docs.dask.org/en/latest/install.html
- ~~ray~~
    - Windows向けに，WSLでもインストールしづらいっぽい？一旦インストールしない
- ~~joblib~~
    - インストール成功か微妙．一旦インストールしない

##### DB接続
- pymssql
    - SQLServer/SQLDB用
- psycopg2-binary
    - PostgreSQL用
    - SQLAlchemyの裏で動いたりする


### NN系
- tensorflow
    - https://www.tensorflow.org/install/pip?hl=ja#3.-install-the-tensorflow-pip-package
- keras
    - https://keras.io/ja/#_2
- ~~chainer~~
    - 開発中止
- torch
    - 下記の関連ライブラリも一緒に入れる
        - torchvision
        - torchaudio
        - torchtext
        - torch-geometric
            - https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
            - https://data.pyg.org/whl/
    - https://pytorch.org/
        - ここで生成されるインストールコマンドを動かすのではなく，下記のwhlリンクを1つずつ探して動かした方が，Pipfile的に安全かも
    - https://download.pytorch.org/whl
        - https://download.pytorch.org/whl/cpu/torch_stable.html
- mxnet
    - https://mxnet.apache.org/versions/1.9.0/get_started?platform=linux&language=python&processor=cpu&environ=pip&
- ~~caffe~~
    - http://caffe.berkeleyvision.org/installation.html
    - どうもGPUが必要なのとインストール手順面倒そうなので一旦インストールしない
- ~~renom~~
    - https://www.renom.jp/packages/renomdl/ja/rsts/installation/installation.html
    - 開発中止？Python3.6までっぽい
- dgl
    - https://www.dgl.ai/pages/start.html
    - https://data.dgl.ai/wheels/repo.html
    - これと一緒にdglgoを入れた方が良さげ
        - https://github.com/dmlc/dgl/tree/master/dglgo
- ~~stellargraph~~
    - https://stellargraph.readthedocs.io/en/stable/README.html#installation
    - Python3.6までっぽく，それ以降ではインストールできそうにない
- jax
    - https://github.com/google/jax#installation
- jaxlib
    - https://jax.readthedocs.io/en/latest/developer.html#building-or-installing-jaxlib

### 機械学習関連の便利・ラッパー系
- scikit-learn
    - https://scikit-learn.org/stable/install.html
- lightgbm
    - https://lightgbm.readthedocs.io/en/latest/Python-Intro.html#install
- pytorch-lightning
    - https://www.pytorchlightning.ai/
- transformers
    - https://huggingface.co/docs/transformers/installation
- ~~pycaret~~
    - https://pycaret.gitbook.io/docs/get-started/installation
    - インストールに時間かかりすぎるので一旦インストールしない
- ~~darts~~
    - https://unit8co.github.io/darts/#quick-install
    - これのインストール前にpystanを入れた方が良さげ
    - でもpystanを入れてもインストール失敗するな…
- optuna
    - https://optuna.org/#installation

### 自然言語処理系
- gensim
    - https://radimrehurek.com/gensim/
    - これと一緒にpythoon-Levenshteinを入れた方が良さげ
- janome
    - https://github.com/mocobeta/janome#install
- ~~mecab-python3~~
    - https://github.com/SamuraiT/mecab-python3#installation
    - これ単体ではたぶん使えない．janomeの方がすぐ使える．
    - 一旦インストールしない
- fasttext
    - https://fasttext.cc/docs/en/python-module.html#installation
- spacy
    - https://spacy.io/usage
- ginza
    - https://github.com/megagonlabs/ginza#2-install-ginza-nlp-library-with-standard-model
    - これと一緒にspacy・ja_ginzaを入れた方が良さげ
- nlplot
    - https://github.com/takapy0210/nlplot#installation

### 画像処理系
- opencv-python
    - https://github.com/opencv/opencv-python#installation-and-usage
- Pillow
    - https://pillow.readthedocs.io/en/stable/installation.html
- mediapipe
    - https://google.github.io/mediapipe/getting_started/python.html#ready-to-use-python-solutions

### 確率的アルゴリズム
- pystan
    - https://github.com/stan-dev/pystan#getting-started
    - gccとclangがいるのか…

### 進化的・遺伝的アルゴリズム
- deap
    - https://github.com/DEAP/deap#installation


### その他
##### 距離系
- python-Levenshtein
    - https://github.com/ztane/python-Levenshtein#installation

##### ライブラリ管理
- pipdeptree


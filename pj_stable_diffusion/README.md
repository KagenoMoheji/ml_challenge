docker-compose on WSL2 with GPU使ってStableDiffusionを遊びたい．

### 手順メモ20230508
1. GPU実行環境構築
    - https://zenn.dev/utahka/articles/ed881a568246f4
    1. Nvidiaドライバをインストール
        - https://developer.nvidia.com/cuda/wsl
        - ドライバのタイプを下記にしてダウンロードしたうえでインストールする  
        ※当時のWindows11に導入しているGPUは「GeForce GTX 1070」．
            - Product Type: GeForce
            - Product Series: GeForce 10 Series
            - Product: GeForce GTX 1070
            - Operating System: Windows11
            - Download Type: Game Ready Driver(GRD)
            - Language: Japanese
        - たぶんNvidia for WSL2も含まれてる
    1. WSL2上でCUDA Toolkitのインストール
        ```
        wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
        sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
        wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb
        sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb
        sudo cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
        sudo apt-get update
        sudo apt-get -y install cuda
        echo "export PATH=\"/usr/local/cuda/bin:$PATH\"" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
        ```
        - https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
            - CUDAをWSLでインストールするコマンドの発行サイト
            - ただnvidia-cuda-toolkitのインストールも含まれているがパスを通すコマンドが無かったので上記に追加した．
                - https://garivinegar.com/cuda-wsl/
                - `sudo apt install nvidia-cuda-toolkit`はバージョン古いのでやるべからず．
        - Nvidiaドライバを最新化させていれば，CUDAも最新バージョンで大丈夫かもしれない
            - 当時はNvidiaのバージョンが「531.79」だったのでCUDA12.1.xのインストールできた
            - ただpytorch2.0.0が当時CUDA11.8までしか対応してなかった(追加情報：python3.8以上，C++17互換のclangコンパイラも語幹として必要)
                - https://pytorch.org/get-started/locally/
                - https://github.com/pytorch/pytorch#prerequisites
                - でも下記サイトでCUDA11.7のpytorchをCUDA12で問題なく動かせているらしいことは書いてある
                    - https://kanpapa.com/today/2023/01/linux-gpu-pc-cuda12-pytorch.html
            - Refs
                - https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
                    - NvidiaドライバとCUDAの対応表
                - https://www.tensorflow.org/install/source?hl=ja#gpu
                    - CUDAとtensorflowの対応表
                    - 当時の最新tensorflow2.12.0はpython3.8.3/CUDA11.8/cuDNN8.6で動く仕様の模様
    1. WSL2上でcuDNNのインストール
        1. 当時のNvidiaドライバ/CUDA/WSL2のUbuntuのバージョンに合わせて「cuDNN8.9.1 for CUDA12.x」の「Local Installer for Ubuntu20.04 x86_64 (Deb)」をインストール
            - https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html
                - 対応表
            - https://developer.nvidia.com/rdp/cudnn-download
                - ダウンロードサイト．ログインが必要かも
        1. (WSL2から`/mnt/c/`でのWindowsファイルシステムのマウントしてある前提で)下記コマンドをWSL2で実行してcuDNNをインストール
            ```
            sudo dpkg -i /mnt/c/Users/<ユーザ名>/Downloads/cudnn-local-repo-ubuntu2004-8.9.1.23_1.0-1_amd64.deb
            ```
            - ダウンロードフォルダにあるdebファイルを使ってインストールしてる
            - なんか一度下記警告文出して失敗する．警告文にあるコマンドを叩いた後にもう一度インストールするとうまくいく
                ```
                The public cudnn-local-repo-ubuntu2004-8.9.1.23 GPG key does not appear to be installed.
                To install the key, run this command:
                sudo cp /var/cudnn-local-repo-ubuntu2004-8.9.1.23/cudnn-local-A9C84908-keyring.gpg /usr/share/keyrings/
                ```
    1. 下記コマンドでWSL2からGPUを検出できているか確認
        ```
        nvidia-smi
        ```
        ```
        nvcc -V
        ```
1. docker-composeでGPU(Nvidia)を使うdockerコンテナ起動
    ```
    version: "3"

    services:
      dev:
        ...
        deploy:
          # ここでGPUの指定
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1 # WARNING: これ設定しないとGPU検知してくれない
                  capabilities: [gpu]
    ```
    - `docker-compose.yml`の上記の項目`resources`でGPUをdockerが検知できるように設定する
    - Refs
        - https://docs.docker.jp/compose/gpu-support.html
        - https://matsuand.github.io/docs.docker.jp.onthefly/compose/gpu-support/
        - https://qiita.com/routerman/items/c5f9d7b6d03e44de6be2
1. dockerコンテナでのtorchのインストール，GPUの動作確認
    - https://stackoverflow.com/questions/63974588/how-to-install-pytorch-with-pipenv-and-save-it-to-pipfile-and-pipfile-lock
    1. https://download.pytorch.org/whl/torch_stable.html
        - python3.11に対応させるため「cu118/torch-2.0.0%2Bcu118-cp311-cp311-linux_x86_64.whl」をダウンロード
        - pipenvでのインストールで`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`のようにオプション「--index-url」が無いためダウンロードしてのインストールの手順を踏まざるを得ない．
        - あれ，`pipenv install --index https://download.pytorch.org/whl/cu118 torch`でいける？なんかすっげー時間かかって中断しちゃったけど…
    1. (whl/targzをダウンロードしてきた場合に実施)Pipfileがある場所で下記コマンドでインストール
        ```
        pipenv run python -m pip install torch-2.0.0%2Bcu118-cp311-cp311-linux_x86_64.whl
        ```
        - 完全なオフラインだと`python -m pip install --no-index --find-links <依存含むライブラリ群を配置しているディレクトリ> <ライブラリのファイル名>`でのインストールになる
    1. 下記スクリプトを実行してTrueが表示されていたらpython on docker on wsl2でGPUが使えていると分かる
        ```
        import torch
        print(torch.cuda.is_available())
        ```
- もしこうなったら
    - 機械学習など重い処理を実行して「RuntimeError: CUDA out of memory.」と処理中断させられる
        - dockerコンテナ(pytorch実行ターミナル)内で`export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128`してから処理を実行してみる．
            - https://www.kageori.com/2022/11/stable-diffusion-web-uiruntimeerror.html
            - https://stealthoptional.com/tech/stable-diffusion-runtime-error-how-to-fix-cuda-out-of-memory-error-in-stable-diffusion/
        - バッチサイズやサンプルサイズを小さくする
        - GPUで動いているプロセスを止めてメモリ解放
            - https://www.tetsumag.com/2022/07/13/283/


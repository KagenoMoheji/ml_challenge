docker-compose on WSL2 with GPU使ってStableDiffusionを遊びたい．
TODO: GPU使うあたり分からん

### Refs
- docker-composeでGPU使う参考
    - https://matsuand.github.io/docs.docker.jp.onthefly/compose/gpu-support/
    - https://qiita.com/naka345/items/eba1870fba589a68847e
        - https://astherier.com/blog/2021/07/windows11-cuda-on-wsl2-setup/
            - ubuntu2004を使ってるので，`cuda-toolkit`ではなく`cuda`のインストールでよさげ
    - https://zenn.dev/okz/articles/83e6f899150b1e
    - https://qiita.com/routerman/items/c5f9d7b6d03e44de6be2
    - https://zenn.dev/holliy/articles/e1ac7f2f806c35
- CUDA周り
    - https://astherier.com/blog/2021/07/windows11-cuda-on-wsl2-setup/
    - https://e-penguiner.com/install-upgrade-specific-cuda/
    - https://tecsingularity.com/cuda/version_conf/
    - https://zenn.dev/gomo/articles/7f6c28d002837c

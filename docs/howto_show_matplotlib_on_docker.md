## 背景
matplotlib on WSLでグラフが表示されない．
たぶんDocker on Linuxでも同じだろう．

## 方法案
- WSLからmatplotlibやるには下記対応が必要そう
    - https://qiita.com/ryoi084/items/c4339996c50c0cf39df4
- `plt.show()`の代わりに，Streamlit・はDash・PyWebIOのいずれかでの描画をし，コンテナからホストへWebサーバ起動して可視化
    - `pj_all_ml_libs/workspace/all_ml_libs/main/proc_study_torch/main_transformer_sigmoid_and_plot_by_streamlit.py`で検証済み

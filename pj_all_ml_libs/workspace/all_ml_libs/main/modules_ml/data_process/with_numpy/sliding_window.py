import numpy as np

def np_sliding_window_view(x, window_size):
    '''
    numpy公式で実装されている移動窓でのローリング配列生成の関数を含め，多くがstrides依存の1次元配列のみ対応で，多次元配列に対するローリングが想定通りの結果にならなかったので自作．
    - Refs
        - strides依存のやつら
            - https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
            - https://zenn.dev/taku227/articles/833455ace8a3aa
            - https://qiita.com/bauer/items/48ef4a57ff77b45244b6
            - https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy/6811241#6811241
    '''
    if x.shape[0] < window_size:
        # numpy配列の長さがwindowサイズより小さい場合
        print("Warning: `x.shape[0]={}` < `window_size={}`".format(x.shape[0], window_size))
        return None
    return np.array([
        x[start_idx:start_idx + window_size] # スライスの後ろのインデックス「未満」になるので-1不要．
        for start_idx in range(len(x) - (window_size - 1)) # ウィンドウ数だけ回す
    ])


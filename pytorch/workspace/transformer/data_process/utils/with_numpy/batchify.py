import numpy as np

def batchify(
    np_X,
    np_y,
    batch_size,
    shuffle = True,
    use_leftovers = True):
    '''
    numpy配列を(必要あればランダムシャッフルして)ミニバッチデータ群に分割して返す．
    なお，分割した結果の残り物の個数がbatch_sizeより少ない場合があるので，異なる次元を配列できないnumpyではなくlistにミニバッチデータを格納して返す．
    - Args
        - np_X:numpy.array: ミニバッチ分割するデータ①
        - np_y:numpy.array: ミニバッチ分割するデータ②
        - batch_size:int: バッチサイズ
        - shuffle:bool: ミニバッチ分割前にシャッフルするか
        - use_leftovers:bool: ミニバッチ分割の最後に，batch_sizeより少なく残ったデータを使うか
    '''
    list_batches_X = []
    list_batches_y = []
    if np_y is not None:
        if len(np_X) != len(np_y):
            raise Exception("Not match the length of 'np_X' and 'np_y'")
        if shuffle:
            p = np.random.permutation(len(np_X))
            np_X, np_y = np_X[p], np_y[p]
        for i in range(-(-len(np_X) // batch_size)):
            if (i == len(np_X) // batch_size) and not use_leftovers:
                continue
            list_batches_X.append(np_X[i * batch_size:i * batch_size + batch_size])
            list_batches_y.append(np_y[i * batch_size:i * batch_size + batch_size])
        return list_batches_X, list_batches_y
    # `np_y = None`の場合はnp_Xに対してのみ上記と同じミニバッチ分割をする
    if shuffle:
        p = np.random.permutation(len(np_X))
        np_X = np_X[p]
    for i in range(-(-len(np_X) // batch_size)):
        if (i == len(np_X) // batch_size) and not use_leftovers:
            continue
        list_batches_X.append(np_X[i * batch_size:i * batch_size + batch_size])
    return list_batches_X, None


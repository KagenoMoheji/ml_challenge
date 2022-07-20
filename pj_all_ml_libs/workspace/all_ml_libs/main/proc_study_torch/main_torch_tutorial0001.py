'''
https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
'''
import inspect
import os
import sys
PYPATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOTPATH = PYPATH + "/./.." # `main`をrootにする
MODNAME = inspect.getfile(inspect.currentframe()).split("/")[-1]
sys.path.append(ROOTPATH)

import numpy as np
import pandas as pd
import torch

# CPU/CUDA環境の確認
print("[CPU or GPU]====================================")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# listからtensorに変換
print("[list to tensor]====================================")
l_data = [[1, 2], [3, 4]]
ts_data = torch.tensor(l_data)
print(ts_data)
print(len(ts_data))
print(ts_data.shape)
print(ts_data.size())
print(type(ts_data))
print(ts_data.dtype)
print(ts_data.device)
# ts_data = ts_data.to("cuda") # CPU版torchでは駄目
# print(ts_data.device)

# numpyからtensorに変換
print("=[numpy to tensor]====================================")
np_data = np.array(l_data)
ts_data = torch.from_numpy(np_data)
print(ts_data)

# 0埋めでtensorを生成
print("=[generate tensor with only zaro]====================================")
ts_data = torch.zeros((5, 2))
print(ts_data)

# ランダム値でtensorを生成
print("=[generate tensor with random int]====================================")
ts_data = torch.rand((2, 3))
print(ts_data)

# インデックスを指定して取得
print("=[get value with index]====================================")
print(ts_data[0])
# スライサー
print("=[slicer]====================================")
print(ts_data[:, 0])
print(ts_data[..., 0])
ts_data[:, 1] = 0
print(ts_data)

# tensorを結合
print("=[join tensors]====================================")
ts_data = torch.cat([ts_data, ts_data], dim = 1) # 行方向に結合
print(ts_data)
ts_data = torch.cat([ts_data, ts_data], dim = 0) # 列方向に結合
print(ts_data)

# 行列計算
print("=[matrix calculation]====================================")
print(ts_data @ ts_data.T)
print(ts_data @ ts_data.T == ts_data.matmul(ts_data.T))

# 配列計算
print("=[array calculation]====================================")
print(ts_data * ts_data)
print(ts_data * ts_data == ts_data.mul(ts_data))
print(ts_data + 20)
print(ts_data + 20 == ts_data.add(20))

# 集計
print("=[aggregate]====================================")
print(ts_data.sum())
print(ts_data.sum().item())

# tensorからnumpyに変換
print("=[tensor to numpy]====================================")
print(ts_data.numpy())
print(ts_data.numpy().shape)
print(ts_data.numpy().dtype)

# tensorからlistに変換
print("=[tensor to list]====================================")
print(ts_data.tolist())

# tensorからpandasに変換
print("=[tensor to pandas]====================================")
pddf_data = pd.DataFrame(ts_data) # .numpy()
print(pddf_data)

# pandasからtensorに変換
print("=[pandas to tensor]====================================")
print(torch.tensor(pddf_data.values)) # .astype(np.float32)
print(torch.tensor(pddf_data.to_numpy())) # dtype = np.float32

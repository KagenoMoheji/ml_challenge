'''
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
'''
import inspect
import os
import sys
PYPATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOTPATH = PYPATH + "/./.." # `main`をrootにする
sys.path.append(ROOTPATH)

import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import streamlit as st

from modules_ml.models.torch_tutorial_nn import NeuralNetwork
from modules_common.myutils.myfigobserver import FigObserver


def main():
    DIR_DATA_DOWNLOADS = "{}/inputs/downloads".format(ROOTPATH)
    DIR_TRAINED_MODELS = "{}/outputs/models".format(ROOTPATH)

    # (我流)モデル(や前処理)パラメータを辞書型で管理
    model_params = {
        "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_clsargs": {},
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 5,
    }

    # データの用意
    ## TODO: ToTensor()以外って何がある？
    data_train = datasets.FashionMNIST(
        root = DIR_DATA_DOWNLOADS,
        train = True,
        download = True,
        transform = ToTensor()
    )
    data_test = datasets.FashionMNIST(
        root = DIR_DATA_DOWNLOADS,
        train = False,
        download = True,
        transform = ToTensor()
    )

    # データをDataloaderに変換
    dataloader_train = DataLoader(data_train, batch_size = model_params["batch_size"])
    dataloader_test = DataLoader(data_test, batch_size = model_params["batch_size"])
    for X, y in dataloader_test:
        print("Shape of X[N, C, M, W]: ", X.shape)
        print("Shape of y: ", {y.shape}, {y.dtype})
        break

    # モデル構築
    print("Using {} device".format(model_params["torch_device"]))
    model = NeuralNetwork(**model_params["model_clsargs"]).to(model_params["torch_device"])
    print(model)
    ## 損失関数にモデルを組み込む
    fn_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = model_params["lr"])

    # ミニバッチ学習をエポックの数だけ周回
    ## エポック＝全データの周回数
    ## ミニバッチデータ＝全データをbatch_sizeに分割した1つのデータ群
    ## したがって，Dataloaderは「ミニバッチデータのまとまり」といえる．
    ## さらに，Datasetは「大元の全データ」といえる．
    ## TODO: エポックごとにミニバッチデータの入れ替えと言った再生成をしないといけないと思うが，自動でやってくれてるの？
    ## というか，変数modelは参照渡しで上書きされてるんだな～…
    plh = st.empty()
    figobserver = FigObserver()
    figobserver.create_ax(ax_name = "train loss", locate = 1)
    figobserver.create_ax(ax_name = "test loss", locate = 3)
    figobserver.create_ax(ax_name = "test accuracy", locate = 4)
    figobserver.print_ax_list()
    with plh.container():
        st.pyplot(figobserver.get_fig())
    losses_train = []
    losses_test = []
    accs_test = []
    for ep in range(model_params["epochs"]):
        print("Epoch {ep}==========================".format(ep = ep + 1))
        # エポック単位での学習済みモデルを作る
        losses_train = train(
            dataloader_train,
            model,
            fn_loss,
            optimizer,
            model_params["torch_device"],
            losses = losses_train)
        figobserver.get_ax("train loss").clear()
        figobserver.get_ax("train loss").plot(losses_train)
        figobserver.get_ax("train loss").set_title("Train Loss")
        # エポック単位で学習したモデルを使ってテストデータでの予測をする
        ## どのエポックの段階の学習済みモデルが最も精度が良いかをここで本来選択する
        losses_test, accs_test = test(
            dataloader_test,
            model,
            fn_loss,
            model_params["torch_device"],
            losses = losses_test,
            accuracies = accs_test)
        figobserver.get_ax("test loss").clear()
        figobserver.get_ax("test loss").plot(losses_test)
        figobserver.get_ax("test loss").set_title("Test Loss")
        figobserver.get_ax("test accuracy").clear()
        figobserver.get_ax("test accuracy").plot(accs_test)
        figobserver.get_ax("test accuracy").set_title("Test Accuracy")
        with plh.container():
            st.pyplot(figobserver.get_fig())
    del figobserver
    # plh.empty()
    print("Finised training.")

    # 学習済みモデルの保存
    dir_output_model = "{}/tutorial0000".format(DIR_TRAINED_MODELS)
    os.makedirs(dir_output_model, exist_ok = True)
    torch.save(model.state_dict(), "{}/model.pth".format(dir_output_model))
    with open("{}/model_params.json".format(dir_output_model), mode = "w", encoding = "utf8") as f:
        f.write(
            json.dumps(
                model_params,
                indent = 4,
                ensure_ascii = False)
        )
    
    # 学習済みモデルを読み込んで，未知データで予測的なことをする
    dir_load_model = dir_output_model
    model_params = None
    with open("{}/model_params.json".format(dir_load_model), mode = "r", encoding = "utf8") as f:
        model_params = json.loads(f.read())
    model = NeuralNetwork(**model_params["model_clsargs"])
    model.load_state_dict(torch.load("{}/model.pth".format(dir_load_model)))
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    # 予測モードにする
    model.eval()
    X, y = data_test[0][0], data_test[0][1]
    with torch.no_grad():
        pred = model(X)
        pred, true = classes[pred[0].argmax(0)], classes[y]
        print("[pred, true]: [{pred}, {true}]".format(pred = pred, true = true))


def train(
    dataloader,
    model,
    fn_loss,
    optimizer,
    torch_device,
    losses = None):
    '''
    学習(訓練)用データで学習
    '''
    if losses is None:
        losses = []
    # 学習データの総数を取得
    size = len(dataloader.dataset)
    ## ちなみにミニバッチデータの個数は下記で取得できる
    # print(len(dataloader))
    # 学習モードにする
    ## TODO: このコードでどの部分が切り替わっている？
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(torch_device), y.to(torch_device)

        # 誤差逆伝搬法で学習
        pred = model(X)
        loss = fn_loss(pred, y)
        ## 誤差逆伝播法
        ### TODO: zero_grad()って何
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            # 100バッチ単位で損失(≒誤差？)と学習に使用したデータ数を表示
            loss, current = loss.item(), batch * len(X)
            losses.append(loss)
            print("loss: {loss:>7f} [{current:>5d}/{size:>5d}]".format(
                loss = loss,
                current = current,
                size = size)
            )
    return losses

def test(
    dataloader,
    model,
    fn_loss,
    torch_device,
    losses = None,
    accuracies = None):
    '''
    テスト用データで予測(性能検証)
    '''
    if losses is None:
        losses = []
    if accuracies is None:
        accuracies = []
    size = len(dataloader.dataset)
    cnt_batches = len(dataloader)
    # 予測モードにする
    ## TODO: このコードでどの部分が切り替わっている？
    model.eval()

    loss, correct = 0, 0
    # TODO: no_grad()って何
    with torch.no_grad():
        # シャッフル無しのミニバッチデータ形式で予測を一括処理する
        for X, y in dataloader:
            X, y = X.to(torch_device), y.to(torch_device)
            pred = model(X)
            # TODO: item()は何を取ってる？
            loss += fn_loss(pred, y).item()
            # TODO: argmax(1)は何？なんのため？
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # 損失の総和をバッチ数で平均取って，予測の損失とする
    loss /= cnt_batches
    losses.append(loss)
    # 正解率の総和をバッチ数で平均取って，予測の正解率とする
    correct /= size
    accuracies.append(correct)
    print("Test Error:\nAccuracy: {corr:>0.1f}%, Avg loss: {loss:>8f}".format(
        corr = 100 * correct,
        loss = loss)
    )
    return losses, accuracies




if __name__ == "__main__":
    main()




'''
[pytorchメモ]

- バッチ予測の予測値を取得する流れ
    - pred
    - pred.detach()
    - pred.detach().squeeze()
    - pred.detach().squeeze().item()
    - pred[0, 0].item() # Tensorのスライスを使った方法
    - pred.detach().numpy()[0, 0] # numpy()を使った方法
'''



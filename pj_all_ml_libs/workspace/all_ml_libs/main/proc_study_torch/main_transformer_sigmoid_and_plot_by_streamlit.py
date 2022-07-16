'''
- Usage
    - streamlit run main_transformer_sigmoid_and_plot_by_streamlit.py -- -m torch_transformer_sigmoid_streamlit
- Refs
    - torch
        - [pyTorchのTensor型とは](https://qiita.com/mathlive/items/241bfb42d852bb801b96#8-2-tensor--ndarray)
    - streamlit
        - [データサイエンティストの飛び道具としてのStreamlit - プロトタイピングをいい感じにする技術](https://tech.jxpress.net/entry/data-app-for-streamlit)
        - [Python: Streamlit を使って手早く WebUI 付きのプロトタイプを作る](https://blog.amedama.jp/entry/streamlit-tutorial)
            - [スクリプトでコマンドライン引数を受け取る](https://blog.amedama.jp/entry/streamlit-tutorial#%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%97%E3%83%88%E3%81%A7%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89%E3%83%A9%E3%82%A4%E3%83%B3%E5%BC%95%E6%95%B0%E3%82%92%E5%8F%97%E3%81%91%E5%8F%96%E3%82%8B)
        - [Streamlit configuration](https://docs.streamlit.io/library/advanced-features/configuration)
        - streamlitでグラフ再描画する方法
            - [st.empty](https://docs.streamlit.io/library/api-reference/layout/st.empty)
            - [Refresh graph/datatable inside loop](https://discuss.streamlit.io/t/refresh-graph-datatable-inside-loop/7804/2)
'''

import inspect
import os
import sys
PYPATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOTPATH = PYPATH + "/./.." # `main`をrootにする
MODNAME = inspect.getfile(inspect.currentframe()).split("/")[-1]
sys.path.append(ROOTPATH)

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import streamlit as st

from modules_common.argparser.ml import get_ml_parser
from modules_ml.models.transformer import TransformerModel


def get_data(batch_size, in_seq_len, out_seq_len):
    i = in_seq_len + out_seq_len
    t = torch.zeros(batch_size, 1).uniform_(0, 20 - i).int()
    b = torch.arange(-10, -10 + i).unsqueeze(0).repeat(batch_size, 1) + t
    s = torch.sigmoid(b.float())
    return s[:, :in_seq_len].unsqueeze(-1), s[:, -out_seq_len:]


def main(dir_models, model_name, is_pred):
    if not is_pred:
        # 学習
        transformer_params = {
            "pytorch_device": "cpu", # torch.device("cuda" if torch.cuda.is_available() else "cpu")
            "enc_seq_len": 6,
            "model_clsargs": {
                "dec_seq_len": 2,
                "input_size": 1,
                "out_seq_len": 1,
                "dim_val": 10,
                "dim_attn": 5,
                "n_heads": 3,
                "n_encoder_layers": 3,
                "n_decoder_layers": 3,
                "dropout_enc": 0.5,
                "dropout_dec": 0.2,
                "dropout_pe": None
            },
            "lr": 0.002,
            "epochs": 10,
            "batch_size": 15
        }
        transformer = TransformerModel(**transformer_params["model_clsargs"]).to(transformer_params["pytorch_device"])
        optimizer = torch.optim.Adam(transformer.parameters(), lr = transformer_params["lr"])
        print(transformer)

        losses = []
        plh = st.empty()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        fig.canvas.draw()
        with plh.container():
            st.pyplot(fig)
        for e in range(transformer_params["epochs"]):
            out = []
            for b in range(-10 - transformer_params["enc_seq_len"], 10 - transformer_params["enc_seq_len"]):
                # バッチごとに勾配を初期化
                optimizer.zero_grad()
                X, y = get_data(transformer_params["batch_size"], transformer_params["enc_seq_len"], transformer_params["model_clsargs"]["out_seq_len"])
                X = X.to(transformer_params["pytorch_device"])
                y = y.to(transformer_params["pytorch_device"])
                # print(X)
                # print(y)
                # モデルを学習モードにしてから入力データを渡す
                transformer.train()
                net_out = transformer(X, training = True)
                # print(net_out.shape, y.shape)
                loss = torch.mean((net_out - y) ** 2)

                loss.backward()
                optimizer.step()

                # print("raw: ", net_out)
                # print("detached: ", net_out.detach())
                # print("squeezed: ", net_out.detach().squeeze())
                # print("unsqueezed: ", net_out.detach().unsqueeze(-1))
                # print("numpyed: ", net_out.detach().numpy())
                # print("listed: ", net_out.detach().numpy().tolist())

                out.append([net_out.detach().numpy(), y])
                losses.append(loss.detach().numpy()) # detachしないとプロットできないので注意

                ax.clear()
                ax.plot(losses)
                ax.set_title("Mean Squared Error")
                fig.canvas.draw()
                with plh.container():
                    st.pyplot(fig)
                plt.pause(0.03)

        dir_output_model = "{0}/{1}".format(dir_models, model_name)
        os.makedirs(dir_output_model, exist_ok = True)
        torch.save(transformer.state_dict(), "{}/model.pth".format(dir_output_model))
        with open("{0}/model_params.json".format(dir_output_model), mode = "w", encoding = "utf8") as f:
            f.write(
                json.dumps(
                    transformer_params,
                    indent = 4,
                    ensure_ascii = False)
            )
        # plh.empty()
    else:
        # 予測
        dir_load_model = "{0}/{1}".format(dir_models, model_name)
        transformer_params = None
        with open("{}/model_params.json".format(dir_load_model), mode = "r", encoding = "utf8") as f:
            transformer_params = json.loads(f.read())
        transformer = TransformerModel(**transformer_params["model_clsargs"]).to(transformer_params["pytorch_device"])
        transformer.load_state_dict(torch.load("{}/model.pth".format(dir_load_model)))

        plh = st.empty()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        fig.canvas.draw()
        with plh.container():
            st.pyplot(fig)

        true = []
        pred = [torch.sigmoid(torch.arange(-10, -3).float()).unsqueeze(-1).numpy().tolist()]
        for i in range(-10, 10, transformer_params["model_clsargs"]["out_seq_len"]):
            true.append([torch.sigmoid(torch.tensor(i).float())])
            q = torch.tensor(pred).float()
            # モデルを予測モードにしてから入力データを渡す
            transformer.eval()
            if transformer_params["model_clsargs"]["out_seq_len"] == 1:
                pred[0].append([transformer(q, training = False).detach().squeeze().numpy().tolist()])
            else:
                for a in transformer(q, training = False).detach().squeeze().numpy().tolist():
                    pred[0].append([a])

            # 下記でプロットをリアルタイム更新できる
            ## たぶん事前に`plt.ion()`しなくてもイケてる…
            ax.clear()
            ax.plot(pred[0], label = "pred")
            ax.plot(true, label = "true")
            ax.set_title("")
            ax.legend(loc = "upper left", frameon = False)
            # fig.canvas.draw()
            with plh.container():
                st.pyplot(fig)
            plt.pause(0.03)


        ax.clear()
        ax.plot(pred[0], label = "pred")
        ax.plot(true, label = "true")
        ax.set_title("")
        ax.legend(loc = "upper left", frameon = False)
        # ノートブックじゃない場所では下記が必要っぽい？
        ## ユーザ操作があるまで完全停止させる(なぜか2画面出るが)
        ### https://stackoverflow.com/a/61804694/15842506
        plt.ioff()
        with plh.container():
            st.pyplot(fig)
        # plh.empty()


if __name__ == "__main__":
    parser = get_ml_parser("Transformerによるシグモイド関数の予測")
    cli_args = parser.parse_args()
    main(
        "{}/outputs/models".format(ROOTPATH),
        cli_args.model,
        cli_args.pred)




import inspect
import os
import sys
PYPATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOTPATH = PYPATH + "/." # `main`をrootにする
sys.path.append(ROOTPATH)

import datetime
import uuid
from diffusers import (
    StableDiffusionPipeline,
    # EulerDiscreteScheduler,
    # DiffusionPipeline,
)
import torch

# 環境変数と定数の設定
## 環境変数
### 「RuntimeError: CUDA out of memory.」を避けるpytorch&cudaの環境変数
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
## 定数
TIMESTAMP_START = datetime.datetime.now()
DEVICE = "cuda"
DIR_INPUT_MODELS = PYPATH + "/inputs/models"
CNT_GEN = 5

# パイプラインの準備
'''
## Huggingfaceのモデルをダウンロードしてパイプラインにする
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_id, 
        subfolder = "scheduler"
    ), 
    torch_dtype = torch.float16
).to(DEVICE)
'''
## Civitaiのsafetensors形式のモデルを読み込んでパイプラインにする
fname_model = "majicmixRealistic_v4.safetensors"
pipe = StableDiffusionPipeline.from_ckpt(
    "{}/{}".format(DIR_INPUT_MODELS, fname_model),
    torch_dtype = torch.float16,
    # use_safetensors = True,
).to(DEVICE)
'''
[*1]StableDiffusionPipelineでできない呪文の75個制限を超えて使える設定が使えると思ったがsafetensorsのモデルの読み込みができなくて断念
pipe = DiffusionPipeline.from_pretrained( 
    "{}/{}".format(DIR_INPUT_MODELS, fname_model),
    custom_pipeline = "lpw_stable_diffusion.py", # [*1]呪文(=token)の75個制限を超えて設定したいときに必要．
    torch_dtype = torch.float16,
    use_safetensors = True,
).to(DEVICE)
'''

# NSWFによる生成画像の黒塗りつぶしを避けるコード．ただ2つ目のリンクにあるライブラリ内でコメントアウトする方法を採用したので使わない．
## https://qiita.com/Limitex/items/10fc8b7f1285d6627fe3
## https://zero-cheese.com/9349/
# if pipe.safety_checker is not None:
#     pipe.safety_checker = lambda images, **kwargs: (images, False)
# 弱いGPUでも処理できるようにする設定を有効化
pipe.enable_attention_slicing()

# プロンプト
prompt = "<ポジティブプロンプト>"
prompt_neg = "<ネガティブプロンプト>"

imgs = pipe(
    prompt,
    negative_prompt = prompt_neg,
    # 8の倍数で高さ幅を設定
    ## ひとまず縦向きモニタ用のアスペクト比にする
    height = 1000,
    width = 568,
    # 生成する画像数
    num_images_per_prompt = CNT_GEN, # https://gammasoft.jp/blog/stable-diffusion-generate-multiple-images/
    # このノイズ除去回数？を増やすと計算量が増えるが画像が綺麗になる．デフォは50．
    num_inference_steps = 30,
    # [*1]呪文(=token)の75個制限を何倍に拡張するか．StableDiffusionPipelineで使えないので断念
    # max_embeddings_multiples = 3,
).images

dir_output_imgs = "{}/outputs/imgs/{}".format(PYPATH, TIMESTAMP_START.strftime("%Y%m%d%H%M%S"))
os.makedirs(dir_output_imgs, exist_ok = True)
for img in imgs:
    fname = "{}/{}.png".format(
        dir_output_imgs,
        uuid.uuid4()
    )
    print("Saved:", fname)
    img.save(fname)

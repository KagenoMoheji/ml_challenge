'''
- Refs
    - https://github.com/eyriewow/merge-models
    - https://self-development.info/%E3%80%90stable-diffusion%E3%80%91%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89%E3%83%A9%E3%82%A4%E3%83%B3%E3%81%AB%E3%82%88%E3%82%8B%E3%83%A2%E3%83%87%E3%83%AB%E7%B5%90%E5%90%88/
    - https://huggingface.co/docs/safetensors/api/torch#safetensors.torch.save_file
'''
import inspect
import os
import sys
PYPATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOTPATH = PYPATH + "/." # `main`をrootにする
sys.path.append(ROOTPATH)

import argparse
import numpy as np
import torch
# from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("model_0", type=str, help="Path to model 0")
parser.add_argument("model_1", type=str, help="Path to model 1")
parser.add_argument("--alpha", type=float, help="Alpha value, optional, defaults to 0.5", default=0.5, required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--outext", type=str, help="Extension of output model.[ckpt|safetensors]", required=False, default="safetensors")
parser.add_argument("--without_vae", action="store_true", help="Do not merge VAE", required=False)
args = parser.parse_args()

theta_0 = None
theta_1 = None
if args.outext == "safetensors":
    # theta_0 = {}
    # theta_1 = {}
    # with safe_open(args.model_0, framework="pt", device=args.device) as f:
    #     for k in f.keys():
    #         theta_0[k] = f.get_tensor(k)
    # with safe_open(args.model_1, framework="pt", device=args.device) as f:
    #     for k in f.keys():
    #         theta_1[k] = f.get_tensor(k)

    # safetensorsてstate_dictのキーそのまま入ってるっぽい？
    ## https://stackoverflow.com/questions/72866756/how-do-i-create-a-model-from-a-state-dict
    theta_0 = load_file(args.model_0, device = args.device)
    theta_1 = load_file(args.model_1, device = args.device)
else:
    model_0 = torch.load(args.model_0, map_location=args.device)
    model_1 = torch.load(args.model_1, map_location=args.device)
    theta_0 = model_0["state_dict"]
    theta_1 = model_1["state_dict"]

output_file = f'{args.output}-{str(args.alpha)[2:] + "0"}'
output_file = f'{output_file}.safetensors' if args.outext == "safetensors" else f'{output_file}.cpkt'

# check if output file already exists, ask to overwrite
if os.path.isfile(output_file):
    print("Output file already exists. Overwrite? (y/n)")
    while True:
        overwrite = input()
        if overwrite == "y":
            break
        elif overwrite == "n":
            print("Exiting...")
            exit()
        else:
            print("Please enter y or n")


for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
    # skip VAE model parameters to get better results(tested for anime models)
    # for anime model，with merging VAE model, the result will be worse (dark and blurry)
    if args.without_vae and "first_stage_model" in key:
        continue
        
    if "model" in key and key in theta_1:
        theta_0[key] = (1 - args.alpha) * theta_0[key] + args.alpha * theta_1[key]

for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
    if "model" in key and key not in theta_0:
        theta_0[key] = theta_1[key]

print("Saving...")

torch.save({"state_dict": theta_0}, output_file)

print("Done!")

# %%
import math
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import display
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid

# %% 

import pickle

paths = {
    # "1Hash_slr": "../logs/blender_chair_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.0005_decay500/loss_vs_time.pkl",
    # "1Hash": "../logs/blender_chair_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.01_decay10/loss_vs_time.pkl",
    # "2Hash": "../logs/blender_chair_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.01_decay10_po2c/loss_vs_time.pkl",
    # --------
    "1Hash": "../logs/blender_chair_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.01_decay10/loss_vs_time.pkl",
    "2Hash": "../logs/blender_chair_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.01_decay10_nhash_2/loss_vs_time.pkl",
    "3Hash": "../logs/blender_chair_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.01_decay10_nhash_3/loss_vs_time.pkl",
    "3HashPool": "../logs/blender_chair_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.01_decay10_nhash_3_pool/loss_vs_time.pkl",
}
# load data
data_dict = {}
for path_key in paths:
    filepath = paths[path_key]
    with open(filepath, "rb") as f:
        data_dict[path_key] = pickle.load(f)

# %% 

# import pdb
# pdb.set_trace()

# plot data
for k in data_dict:
    plt.plot(data_dict[k]["psnr"][1:200][::2], label=k)
    # plt.plot(data_dict[k]["psnr"][1:200][::2], label=k)
    # plt.plot(data_dict[k]["time"][1:200][::2], label=k)
    # plt.plot(data_dict[k]["losses"][1:200][::2], label=k)

plt.legend()
plt.show()
# plt.savefig('./loss_plot.png')

# %%

import pickle
import matplotlib.pyplot as plt
import pdb

paths = {
         #"Vanilla HighLR": "../logs/blender_chair_posXYZ_posVIEW_fine1024_log2T19_lr0.01_decay100/loss_vs_time.pkl", \
         "Hashed Fast": "../logs/blender_chair_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.01_decay100/loss_vs_time.pkl", \
         "Hashed Superfast": "../logs/blender_chair_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.01_decay10/loss_vs_time.pkl", \
         "Vanilla SlowLR": "../logs/blender_chair_posXYZ_posVIEW_fine1024_log2T19_lr0.0005_decay500/loss_vs_time.pkl"}

# load data
data_dict = {}
for path_key in paths:
    filepath = paths[path_key]
    with open(filepath, "rb") as f:
        data_dict[path_key] = pickle.load(f)

# plot data
#for k in data_dict:
for k in data_dict:
    plt.plot(data_dict[k]["psnr"][1:200][::2], label=k)

plt.legend()
plt.show()

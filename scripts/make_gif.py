from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import os
import imageio
import pdb
import numpy as np

image_idx = "000"

paths = {
         "Hashed": "../logs/blender_hotdog_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.01_decay10"}

for path_name, log_path in paths.items():
    folders = [name for name in os.listdir(log_path) if name.startswith("renderonly_path_")]
    folders.sort()
    images = []
    writer = imageio.get_writer(os.path.join(log_path, 'convergence.mp4'), fps=2)
    for i, folder in enumerate(folders):
        if i>50:
            break
        img = Image.open(os.path.join(log_path, folder, image_idx + ".png"))
        font = ImageFont.truetype("arial.ttf", 30)
        ImageDraw.Draw(
            img  # Image
        ).text(
            (0, 0),  # Coordinates
            'Iter: '+str(int(folder[-6:])),  # Text
            (0, 0, 0),  # Color
            font=font
        )
        images.append(img)
        writer.append_data(np.array(img))
    pdb.set_trace()
    writer.close()
    #imageio.mimsave(os.path.join(log_path, 'convergence_dur025.gif'), images, duration=0.25)

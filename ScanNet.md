# ScanNet Instructions

I personally found it a bit tricky to setup the ScanNet dataset the first time I tried it. So, I am compiling some notes/instructions on how to do it in case someone finds it useful.

### 1. Dataset download

To download ScanNet data and its labels, follow the instructions [here](https://github.com/ScanNet/ScanNet). Basically, fill out the ScanNet Terms of Use agreement and email it to [scannet@googlegroups.com](mailto:scannet@googlegroups.com). You will receive a download link to the dataset. Download the dataset and unzip it.

### 2. Use [SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python) to extract RGB-D and camera data
Use the `reader.py` script as follows for each scene you want to work with:
``` 
python reader.py --filename [.sens file to export data from] --output_path [output directory to export data to] 
	  Options:
	  --export_depth_images: export all depth frames as 16-bit pngs (depth shift 1000)
	  --export_color_images: export all color frames as 8-bit rgb jpgs
	  --export_poses: export all camera poses (4x4 matrix, camera to world)
	  --export_intrinsics: export camera intrinsics (4x4 matrix) 
```

### 3. Then, use this [script](https://github.com/zju3dv/object_nerf/blob/main/data_preparation/scannet_sens_reader/convert_to_nerf_style_data.py) to convert the data to NeRF-style format. For instructions, see Step 1 [here](https://github.com/zju3dv/object_nerf/tree/main/data_preparation).
1. The transformation matrices (`c2w`) in the generated transforms_xxx.json will be in SLAM / OpenCV format (xyz -> right down forward). You need to change to NDC format (xyz -> right up back) in the dataloader for training with NeRF convention.
2. For example, see the conversion done [here](https://github.com/cvg/nice-slam/blob/7af15cc33729aa5a8ca052908d96f495e34ab34c/src/utils/datasets.py#L205).

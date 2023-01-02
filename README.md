# HashNeRF-pytorch
[Instant-NGP](https://github.com/NVlabs/instant-ngp) recently introduced a Multi-resolution Hash Encoding for neural graphics primitives like [NeRFs](https://www.matthewtancik.com/nerf). The original NVIDIA implementation mainly in C++/CUDA, based on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), can train NeRFs upto 100x faster!

This project is a **pure PyTorch** implementation of [Instant-NGP](https://github.com/NVlabs/instant-ngp), built with the purpose of enabling AI Researchers to play around and innovate further upon this method.

This project is built on top of the super-useful [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch) implementation.

## Convergence speed w.r.t. Vanilla NeRF
**HashNeRF-pytorch** (left) vs [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch) (right):

https://user-images.githubusercontent.com/8559512/154065666-f2eb156c-333c-4de4-99aa-8aa15a9254de.mp4

After training for just 5k iterations (~10 minutes on a single 1050Ti), you start seeing a _crisp_ chair rendering. :)

# Instructions
Download the nerf-synthetic dataset from here: [Google Drive](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi).

To train a `chair` HashNeRF model:
```
python run_nerf.py --config configs/chair.txt --finest_res 512 --log2_hashmap_size 19 --lrate 0.01 --lrate_decay 10
```

To train for other objects like `ficus`/`hotdog`, replace `configs/chair.txt` with `configs/{object}.txt`:

![hotdog_ficus](https://user-images.githubusercontent.com/8559512/154066554-d3656d4a-1738-427c-982d-3ef4e4071969.gif)

## Extras
The code-base has additional support for:
* Total Variation Loss for smoother embeddings (use `--tv-loss-weight` to enable)
* Sparsity-inducing loss on the ray weights (use `--sparse-loss-weight` to enable)

## ScanNet dataset support
The repo now supports training a NeRF model on a scene from the ScanNet dataset. I personally found setting up the ScanNet dataset to be a bit tricky. Please find some instructions/notes in [ScanNet.md](ScanNet.md).


## TODO:
* Voxel pruning during training and/or inference
* Accelerated ray tracing, early ray termination


# Citation
Kudos to [Thomas Müller](https://tom94.net/) and the NVIDIA team for this amazing work, that will greatly help accelerate Neural Graphics research:
```
@article{mueller2022instant,
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    journal = {arXiv:2201.05989},
    year = {2022},
    month = jan
}
```

Also, thanks to [Yen-Chen Lin](https://yenchenlin.me/) for the super-useful [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch):
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```

If you find this project useful, please consider to cite:
```
@misc{bhalgat2022hashnerfpytorch,
  title={HashNeRF-pytorch},
  author={Yash Bhalgat},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yashbhalgat/HashNeRF-pytorch/}},
  year={2022}
}
```

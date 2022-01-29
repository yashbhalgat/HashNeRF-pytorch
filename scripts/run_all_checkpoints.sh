for i in logs/blender_chair_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.01_decay100_14_24_28_01_2022/*.tar; do
    CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 100 --render_only --ft_path $i
done

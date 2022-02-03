for i in logs/blender_hotdog_hashXYZ_sphereVIEW_fine1024_log2T19_lr0.01_decay10/*.tar; do
    CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/hotdog.txt --finest_res 1024 --lr 0.01 --lr_decay 10 --render_only --ft_path $i
done

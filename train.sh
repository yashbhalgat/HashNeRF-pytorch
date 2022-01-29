# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/chair.txt --finest_res 1024
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/chair.txt --finest_res 1024 --i_embed_views 0
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 100
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/chair.txt --finest_res 1024 --log2_hashmap_size 14 

CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/chair.txt --finest_res 1024 
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 100
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/chair.txt --finest_res 1024 --i_embed 0 --i_embed_views 0
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/chair.txt --finest_res 1024 --i_embed 0 --i_embed_views 0 --lrate 0.01 --lrate_decay 100

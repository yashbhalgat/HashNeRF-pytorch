# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/chair.txt --finest_res 1024
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/chair.txt --finest_res 1024 --i_embed_views 0
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 100
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/chair.txt --finest_res 1024 --log2_hashmap_size 14 

# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/chair.txt --finest_res 1024 
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 100
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/chair.txt --finest_res 1024 --i_embed 0 --i_embed_views 0
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/chair.txt --finest_res 1024 --i_embed 0 --i_embed_views 0 --lrate 0.01 --lrate_decay 100

# # Higher LR, currently recommended
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --num_hashes 2

# # Po2C
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/chair.txt --finest_res 1024 --wandb
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --num_hashes 2 --wandb
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --num_hashes 3 --wandb
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --num_hashes 2 --pool_over_hashes --wandb
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --num_hashes 3 --pool_over_hashes --wandb

# # Sweep 
# CUDA_VISIBLE_DEVICES=0 python run_nerf.py --wandb --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash yash --seed 43 --expname blender_chair_yash_hash_seed43
# CUDA_VISIBLE_DEVICES=1 python run_nerf.py --wandb --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash yash --expname blender_chair_yash_hash
# CUDA_VISIBLE_DEVICES=2 python run_nerf.py --wandb --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash ngp --expname blender_chair_ngp_hash
# CUDA_VISIBLE_DEVICES=3 python run_nerf.py --wandb --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash nonhash --expname blender_chair_non_hash
# CUDA_VISIBLE_DEVICES=4 python run_nerf.py --wandb --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash yash --num_hashes 2 --expname blender_chair_yash_hash_2_hash
# CUDA_VISIBLE_DEVICES=5 python run_nerf.py --wandb --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash ngp --num_hashes 2 --expname blender_chair_ngp_hash_2_hash

# # Debug parallel work
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash ngp --num_hashes 2 --expname debug
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash parallel_ngp --num_hashes 2 --expname debug

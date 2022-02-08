# Example usage:
# ./scripts/launch-tmux.sh

## Session name
NAME=hashnerf_runs_1

## Create tmux session
tmux new-session -d -s ${NAME}

## Create the windows on which each node or .launch file is going to run
tmux send-keys -t ${NAME} 'tmux new-window -n WIN0 ' ENTER
tmux send-keys -t ${NAME} 'tmux new-window -n WIN1 ' ENTER
tmux send-keys -t ${NAME} 'tmux new-window -n WIN2 ' ENTER
tmux send-keys -t ${NAME} 'tmux new-window -n WIN3 ' ENTER
tmux send-keys -t ${NAME} 'tmux new-window -n WIN4 ' ENTER
tmux send-keys -t ${NAME} 'tmux new-window -n WIN5 ' ENTER
# tmux send-keys -t ${NAME} 'tmux new-window -n WIN6 ' ENTER
# tmux send-keys -t ${NAME} 'tmux new-window -n WIN7 ' ENTER

## Send commands to each window
tmux send-keys -t ${NAME} "tmux send-keys -t WIN0 'sleep 1; CUDA_VISIBLE_DEVICES=0 python run_nerf.py --wandb --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash yash --seed 43 --expname blender_chair_yash_hash_seed43' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN1 'sleep 1; CUDA_VISIBLE_DEVICES=1 python run_nerf.py --wandb --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash yash --expname blender_chair_yash_hash' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN2 'sleep 1; CUDA_VISIBLE_DEVICES=2 python run_nerf.py --wandb --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash ngp --expname blender_chair_ngp_hash' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN3 'sleep 1; CUDA_VISIBLE_DEVICES=3 python run_nerf.py --wandb --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash nonhash --expname blender_chair_non_hash' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN4 'sleep 1; CUDA_VISIBLE_DEVICES=4 python run_nerf.py --wandb --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash yash --num_hashes 2 --expname blender_chair_yash_hash_2_hash' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN5 'sleep 1; CUDA_VISIBLE_DEVICES=5 python run_nerf.py --wandb --config configs/chair.txt --finest_res 1024 --lrate 0.01 --lrate_decay 10 --which_hash ngp --num_hashes 2 --expname blender_chair_ngp_hash_2_hash' ENTER" ENTER
# tmux send-keys -t ${NAME} "tmux send-keys -t WIN6 './my_command.sh' ENTER" ENTER
# tmux send-keys -t ${NAME} "tmux send-keys -t WIN7 './my_command.sh' ENTER" ENTER

## Start a new line on window 0
tmux send-keys -t ${NAME} ENTER

## Attach to session
# tmux send-keys -t ${NAME} "tmux select-window -t 1" ENTER
# tmux send-keys -t ${NAME} "tmux send-keys 'nvidia-smi -l 60' ENTER" ENTER
# tmux attach -t ${NAME}
echo "Launched ${NAME}"
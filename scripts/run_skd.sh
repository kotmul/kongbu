export WANDB_API_KEY=""

CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file ./accelerate_configs/zero3.yaml --num_processes=1 \
    ./run_skd.py --config-name skd.yaml
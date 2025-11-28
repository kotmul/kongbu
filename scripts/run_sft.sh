export WANDB_API_KEY=""

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./accelerate_configs/zero3.yaml --num_processes=1 \
    ./run_sft.py --config-name sft.yaml
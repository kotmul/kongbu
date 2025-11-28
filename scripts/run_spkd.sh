export WANDB_API_KEY=""

accelerate launch --config_file ./accelerate_configs/zero3.yaml --num_processes=2 \
    ./run_spkd.py --config-name spkd.yaml
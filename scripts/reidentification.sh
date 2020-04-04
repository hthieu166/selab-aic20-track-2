#!/usr/bin/env bash
source "scripts/master_env.sh"

python main.py \
    --gpu_id $GPUID \
    -w $N_WORKERS \
    --dataset_cfg "./configs/dataset_cfgs/aic20_vehicle_reid.yaml" \
    --model_cfg   "./configs/model_cfgs/aic20_vehicle_reid.yaml" \
    --train_cfg   "./configs/train_cfgs/aic20_vehicle_reid.yaml" \
    --logdir      "logs/aic20_vehicle_reid" \
    --log_fname   "logs/aic20_vehicle_reid/stdout.log" \
    --train_mode  "from_scratch" \
    --is_training false \
    --pretrained_model_path "logs/aic20_vehicle_reid/best.model" \
    --output      "outputs/aic20_vehicle_reid/"


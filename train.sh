#!/bin/bash

# Training script with Multi-View features enabled

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=29501 \
main_train.py \
    --world_size 1 \
    --batch_size 16 \
    --data_path "/root/autodl-tmp/my_dataset/EITL_train.json" \
    --epochs 200 \
    --lr 2e-4 \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --pretrain_path "/root/autodl-tmp/SparseViT-main/checkpoint/train/pretrain/uniformer_base_ls_in1k.pth" \
    --test_data_path "/root/autodl-tmp/my_dataset/EITL_train.json" \
    --warmup_epochs 4 \
    --output_dir ./output_dir_multiview/ \
    --log_dir ./output_dir_multiview/  \
    --accum_iter 1 \
    --seed 42 \
    --test_period 4 \
    --num_workers 8 \
    --model_type single \
    --use_view_aware \
    --view_alpha 1.0 \
    --sparse_topk 64 \
    --fusion_type attention \
#    --resume "/root/autodl-tmp/SparseViT-main/output_dir_multiview/checkpoint-latest.pth"
    2> train_multiview_error.log 1>train_multiview_log.log

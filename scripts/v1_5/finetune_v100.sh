#!/bin/bash

    deepspeed \
    dc/train/train_xformers.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./ckpt/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./instruction_tuning/llava_v1_5_mix665k.json \
    --image_folder ./train/instruction_tuning \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints_stage1/DenseConnector-v1.5-7b-Pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_dense_connector_type dci \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir ./checkpoints_stage2/DenseConnector-v1.5-7b-FineTuning \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --fp16 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

#!/bin/bash

CKPT="llava-v1.5-7b"
CONFIG="llava/eval/MMMU/eval/configs/llava1.5.yaml"

python dc/eval/MMMU/eval/run_llava.py \
        --data_path ./cache_dir/MMMU___mmmu \
        --config_path $CONFIG \
        --model_path liuhaotian/llava-v1.5-13b \
        --answers-file ./playground/data/eval/MMMU/answers/$CKPT/$CKPT.jsonl \
        --split "validation" \
        --conv-mode vicuna_v1


output_file=./playground/data/eval/MMMU/answers/$CKPT/$CKPT.jsonl


python dc/eval/MMMU/eval/eval.py --result_file $output_file --output_path ./playground/data/eval/MMMU/$CKPT/val.json



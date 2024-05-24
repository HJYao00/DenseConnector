#!/bin/bash

CKPT="DenseConnector-v1.5-7B"
CONFIG="dc/eval/MMMU/eval/configs/llava1.5.yaml"

python dc/eval/MMMU/eval/run_llava.py \
        --data_path ./cache_dir/MMMU___mmmu \
        --config_path $CONFIG \
        --model_path DenseConnector-v1.5-7B \
        --answers-file ./playground/data/eval/MMMU/answers/$CKPT/$CKPT.jsonl \
        --split "validation" \
        --conv-mode vicuna_v1


output_file=./playground/data/eval/MMMU/answers/$CKPT/$CKPT.jsonl


python dc/eval/MMMU/eval/eval.py --result_file $output_file --output_path ./playground/data/eval/MMMU/$CKPT/val.json



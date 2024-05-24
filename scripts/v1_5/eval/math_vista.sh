#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="DenseConnector-v1.5-7B"
OPENAIKEY=""
OPENAIBASE=""

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m dc.eval.model_math_vista \
        --model-path DenseConnector-v1.5-7B \
        --question-file ./playground/data/eval/MathVista/testmini.json \
        --image-folder ./MathVista \
        --answers-file ./playground/data/eval/MathVista/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/MathVista/answers/$CKPT/merge.jsonl
score_file=./playground/data/eval/MathVista/answers/$CKPT/score.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/MathVista/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python dc/eval/MathVista/extract_answer.py \
    --output_file $output_file \
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE

python dc/eval/MathVista/calculate_score.py \
    --output_file $output_file \
    --score_file $score_file \
    --gt_file /root/paddlejob/workspace/env_run/mllm_data/MathVista/annot_testmini.json

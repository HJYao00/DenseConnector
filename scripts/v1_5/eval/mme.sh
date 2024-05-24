#!/bin/bash

python -m dc.eval.model_vqa_loader \
    --model-path DenseConnector-v1.5-7B \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/DenseConnector-v1.5-7B.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment DenseConnector-v1.5-7B

cd eval_tool

python calculation.py --results_dir answers/DenseConnector-v1.5-7B

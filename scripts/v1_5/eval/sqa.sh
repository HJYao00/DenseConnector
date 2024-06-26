#!/bin/bash

python -m dc.eval.model_vqa_science \
    --model-path DenseConnector-v1.5-7B \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/DenseConnector-v1.5-7B.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python dc/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/DenseConnector-v1.5-7B.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/DenseConnector-v1.5-7B_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/DenseConnector-v1.5-7B_result.json

#!/bin/bash

python -m dc.eval.model_vqa_loader \
    --model-path DenseConnector-v1.5-7B \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/DenseConnector-v1.5-7B.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python dc/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/DenseConnector-v1.5-7B.jsonl

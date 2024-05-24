#!/bin/bash

python -m dc.eval.model_vqa \
    --model-path DenseConnector-v1.5-7B \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/DenseConnector-v1.5-7B.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/DenseConnector-v1.5-7B.jsonl \
    --dst ./playground/data/eval/mm-vet/results/DenseConnector-v1.5-7B.json


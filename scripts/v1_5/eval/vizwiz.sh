#!/bin/bash

python -m dc.eval.model_vqa_loader \
    --model-path DenseConnector-v1.5-7B \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/DenseConnector-v1.5-7B.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/DenseConnector-v1.5-7B.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/DenseConnector-v1.5-7B.json

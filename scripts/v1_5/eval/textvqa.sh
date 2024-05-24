#!/bin/bash

python -m dc.eval.model_vqa_loader \
    --model-path DenseConnector-v1.5-7B \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/DenseConnector-v1.5-7B.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m dc.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/DenseConnector-v1.5-7B.jsonl

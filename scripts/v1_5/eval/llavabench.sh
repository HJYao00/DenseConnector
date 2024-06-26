#!/bin/bash

python -m dc.eval.model_vqa \
    --model-path DenseConnector-v1.5-7B \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/DenseConnector-v1.5-7B.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python dc/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule dc/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/DenseConnector-v1.5-7B.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/DenseConnector-v1.5-7B.jsonl

python dc/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/DenseConnector-v1.5-7B.jsonl

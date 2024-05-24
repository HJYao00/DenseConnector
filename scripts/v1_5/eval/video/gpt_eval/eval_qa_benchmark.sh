gpt_version="gpt-3.5-turbo-0301" #"gpt-3.5-turbo-0613" "gpt-3.5-turbo-0125"
output_name="llava-v1.5-7b_u4FRS"
pred_dir="Video_Benchmark/${output_name}"
output_dir="Video_Benchmark/${output_name}/${gpt_version}"
api_key="sk-xxx" 
num_tasks=32


# Run the "correctness" evaluation script
python3 scripts/gpt_eval/evaluate_benchmark_1_correctness.py \
    --pred_path "${pred_dir}/generic.jsonl" \
    --output_dir "${output_dir}/correctness_pred" \
    --output_json "${output_dir}/correctness_results.json" \
    --api_key ${api_key} \
    --gpt_version ${gpt_version} \
    --num_tasks ${num_tasks}


# Run the "detailed orientation" evaluation script
python3 scripts/gpt_eval/evaluate_benchmark_2_detailed_orientation.py \
    --pred_path "${pred_dir}/generic.jsonl" \
    --output_dir "${output_dir}/detailed_eval" \
    --output_json "${output_dir}/detailed_orientation_results.json" \
    --api_key ${api_key} \
    --gpt_version ${gpt_version} \
    --num_tasks ${num_tasks}


# Run the "contextual understanding" evaluation script
python3 scripts/gpt_eval/evaluate_benchmark_3_context.py \
    --pred_path "${pred_dir}/generic.jsonl" \
    --output_dir "${output_dir}/context_eval" \
    --output_json "${output_dir}/contextual_understanding_results.json" \
    --api_key ${api_key} \
    --gpt_version ${gpt_version} \
    --num_tasks ${num_tasks}


# Run the "temporal understanding" evaluation script
python3 scripts/gpt_eval/evaluate_benchmark_4_temporal.py \
    --pred_path "${pred_dir}/temporal.jsonl" \
    --output_dir "${output_dir}/temporal_eval" \
    --output_json "${output_dir}/temporal_understanding_results.json" \
    --api_key ${api_key} \
    --gpt_version ${gpt_version} \
    --num_tasks ${num_tasks}    


# Run the "consistency" evaluation script
python3 scripts/gpt_eval/evaluate_benchmark_5_consistency.py \
    --pred_path "${pred_dir}/consistency.jsonl" \
    --output_dir "${output_dir}/consistency_eval" \
    --output_json "${output_dir}/consistency_results.json" \
    --api_key ${api_key} \
    --gpt_version ${gpt_version} \
    --num_tasks ${num_tasks}    


echo "All evaluations completed!"
gpt_version="gpt-3.5-turbo-0301" #"gpt-3.5-turbo-0613" "gpt-3.5-turbo-0125"
output_name="llava-v1.5-7b_u4FRS"
pred_path="MSVD_Zero_Shot_QA/${output_name}/merge.jsonl"
output_dir="MSVD_Zero_Shot_QA/${output_name}/${gpt_version}"
output_json="MSVD_Zero_Shot_QA/${output_name}/results_${gpt_version}.json"
api_key="sk-xxx" 
num_tasks=25



python3 scripts/gpt_eval/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --gpt_version ${gpt_version} \
    --num_tasks ${num_tasks}
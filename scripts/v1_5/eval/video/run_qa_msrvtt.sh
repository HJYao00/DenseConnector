CKPT_NAME="DenseConnector-v1.5-7B"
num_frames=8
model_path="./ckpt/DenseConnector-v1.5-7B"
GPT_Zero_Shot_QA="./path_to_data/GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/videos/all"
gt_file_question="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/test_a.json"
output_dir="output/MSRVTT_Zero_Shot_QA/${CKPT_NAME}_u${num_frames}FRS_D2_tk128_tem0_pool"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 dc/eval/run_inference_qa.py \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --model_name ${model_path} \
      --num_chunks $CHUNKS \
      --num_frames $num_frames \
      --conv-mode vicuna_v1 \
      --temperature 0 \
      --use_pool \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done

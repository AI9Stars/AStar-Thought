thought_begin_tag="<|begin_of_thought|>"
thought_end_tag="<|end_of_thought|>"
solution_begin_tag="<|begin_of_solution|>"
solution_end_tag="<|end_of_solution|>"

alpha=0.5
beta=0.1
min_search_steps=5
max_search_steps=20
device_map="0,1,2,3,4,5,6,7"

scorer_model_path="openai-community/gpt2"
validator_model_path="simplescaling/s1.1-32B"
data_path="<your s1K-1.1 data path>"

CUDA_VISIBLE_DEVICES="${device_map}" \
    python long_cot_compress.py \
    --scorer_model_path "${scorer_model_path}" \
    --validator_model_path "${validator_model_path}" \
    --data_path "${data_path}" \
    --cache_path "./res/cache/s1K-1.1-bis.jsonl" \
    --output_path "./res/data/s1K-1.1-compressed.jsonl" \
    --scorer_works_num 32 \
    --scorer_device_map "${device_map}" \
    --validator_device_map "${device_map}" \
    --thought_begin_tag "${thought_begin_tag}" \
    --thought_end_tag "${thought_end_tag}" \
    --solution_begin_tag "${solution_begin_tag}" \
    --solution_end_tag "${solution_end_tag}" \
    --alpha ${alpha} \
    --beta ${beta} \
    --min_search_steps ${min_search_steps} \
    --max_search_steps ${max_search_steps} \
    --load_s1k
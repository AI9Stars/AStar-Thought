model_path=Qwen/Qwen3.5-0.8B
data_path=TeichAI/deepseek-v3.2-speciale-openr1-math-3k/deepseek-v3.2-speciale-openr1-math-3k.jsonl

### The compression rate is inversely proportional to the angle threshold.
### Reduce the angle threshold: retain the thinking steps more strictly, with a higher compression rate;
### Increase the angle threshold: retain the thinking steps more leniently, with a lower compression rate.
pca_angle_threshold=90 # [0,180]

device_map="0,1,2,3,4,5,6,7"
CUDA_VISIBLE_DEVICES="${device_map}" \
    python long_cot_compress_v2.py \
    --model_path "${model_path}" \
    --data_path "${data_path}" \
    --device_map "${device_map}" \
    --works_num 8 \
    --trajectory_pca_cache_path "./saves/cache/trajectory-pca.jsonl" \
    --output_path "./saves/data/AStar-Thought-V2-OpenR1-Math-3k/train.jsonl" \
    --thought_begin_tag "<think>" \
    --thought_end_tag "</think>" \
    --pca_angle_threshold "${pca_angle_threshold}"
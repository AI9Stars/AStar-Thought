export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model=xxang/AStar-Thought-V2-Qwen3.6-27B
task=aime24 # aime25 aime26 arc_c gpqa_diamond
max_tokens=81920
max_latent_count=4
max_latent_len=32

python -m astarthought.evals.cli evaluate \
    --task ${task} \
    --model ${model} \
    --backend vllm \
    --backend-args "tensor_parallel_size=8,gpu_memory_utilization=0.8,reasoning_parser=qwen3,hf_overrides={'max_latent_count':${max_latent_count},'max_latent_len':${max_latent_len}}" \
    --sampling-params temperature=1.0,top_p=0.95,max_tokens=${max_tokens} \
    --result-dir ./saves/result \
    --overwrite \
    --batch-size 512 \
model="your model path here"
task="math500" # "amc23" "olympiadbench_math_en" "gsm8k"
max_tokens=512 # 1024 2048 4096

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
python -m astarthought.evals.cli evaluate \
    --task ${task} \
    --model ${model} \
    --backend vllm \
    --backend-args tensor_parallel_size=8 \
    --sampling-params temperature=0.6,top_p=0.95,max_tokens=${max_tokens} \
    --result-dir ./res/eval \
    --overwrite
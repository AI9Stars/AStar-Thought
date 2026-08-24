dataset=AStar-Thought-s1K-1.1
model=/home/wangshuo/wangshuo01/models/Qwen/Qwen3.6-27B
template=qwen3_5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
llamafactory-cli train \
    --model_name_or_path $model \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json \
    --dataset $dataset \
    --template $template \
    --cutoff_len 20480 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir ../saves/model/AStarThought-Qwen3.6-27B \
    --logging_steps 1 \
    --save_strategy epoch \
    --plot_loss \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.0e-5 \
    --num_train_epochs 3.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --seed 42 \
    --report_to none \
    --run_name AStarThought-Qwen3.6-27B \
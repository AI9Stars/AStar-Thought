DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 
  "

torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed zero3_config.json \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn \
    --model_name_or_path Qwen/Qwen2.5-Coder-14B-Instruct \
    --dataset qwq \
    --template qwen \
    --finetuning_type full \
    --lora_target q_proj,v_proj\
    --output_dir qwen_qwq \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps1 \
    --cutoff_len 8192 \
    --save_steps 10 \
    --plot_loss \
    --num_train_epochs3  \
    --bf16 \
    --flash_attn fa2 \

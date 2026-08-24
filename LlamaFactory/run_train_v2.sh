dataset=AStarThought-V2-OpenR1-Math-3k
model=/home/wangshuo/wangshuo01/models/Qwen/Qwen3.6-27B
template=qwen3_5

### A*-Thought-V2 hyperparameters
latent_begin_token="<latent>"
latent_end_token="</latent>"
latent_tokens="<latent_1>,<latent_2>,<latent_3>,<latent_4>,<latent_5>,<latent_6>,<latent_7>,<latent_8>,<latent_9>,<latent_10>,<latent_11>,<latent_12>,<latent_13>,<latent_14>,<latent_15>,<latent_16>,<latent_17>,<latent_18>,<latent_19>,<latent_20>,<latent_21>,<latent_22>,<latent_23>,<latent_24>,<latent_25>,<latent_26>,<latent_27>,<latent_28>,<latent_29>,<latent_30>,<latent_31>,<latent_32>,<latent_33>,<latent_34>,<latent_35>,<latent_36>,<latent_37>,<latent_38>,<latent_39>,<latent_40>,<latent_41>,<latent_42>,<latent_43>,<latent_44>,<latent_45>,<latent_46>,<latent_47>,<latent_48>,<latent_49>,<latent_50>,<latent_51>,<latent_52>,<latent_53>,<latent_54>,<latent_55>,<latent_56>,<latent_57>,<latent_58>,<latent_59>,<latent_60>,<latent_61>,<latent_62>,<latent_63>,<latent_64>,<latent_65>,<latent_66>,<latent_67>,<latent_68>,<latent_69>,<latent_70>,<latent_71>,<latent_72>,<latent_73>,<latent_74>,<latent_75>,<latent_76>,<latent_77>,<latent_78>,<latent_79>,<latent_80>,<latent_81>,<latent_82>,<latent_83>,<latent_84>,<latent_85>,<latent_86>,<latent_87>,<latent_88>,<latent_89>,<latent_90>,<latent_91>,<latent_92>,<latent_93>,<latent_94>,<latent_95>,<latent_96>,<latent_97>,<latent_98>,<latent_99>,<latent_100>,<latent_101>,<latent_102>,<latent_103>,<latent_104>,<latent_105>,<latent_106>,<latent_107>,<latent_108>,<latent_109>,<latent_110>,<latent_111>,<latent_112>,<latent_113>,<latent_114>,<latent_115>,<latent_116>,<latent_117>,<latent_118>,<latent_119>,<latent_120>,<latent_121>,<latent_122>,<latent_123>,<latent_124>,<latent_125>,<latent_126>,<latent_127>,<latent_128>,<latent_129>,<latent_130>,<latent_131>,<latent_132>,<latent_133>,<latent_134>,<latent_135>,<latent_136>,<latent_137>,<latent_138>,<latent_139>,<latent_140>,<latent_141>,<latent_142>,<latent_143>,<latent_144>,<latent_145>,<latent_146>,<latent_147>,<latent_148>,<latent_149>,<latent_150>,<latent_151>,<latent_152>,<latent_153>,<latent_154>,<latent_155>,<latent_156>,<latent_157>,<latent_158>,<latent_159>,<latent_160>,<latent_161>,<latent_162>,<latent_163>,<latent_164>,<latent_165>,<latent_166>,<latent_167>,<latent_168>,<latent_169>,<latent_170>,<latent_171>,<latent_172>,<latent_173>,<latent_174>,<latent_175>,<latent_176>,<latent_177>,<latent_178>,<latent_179>,<latent_180>,<latent_181>,<latent_182>,<latent_183>,<latent_184>,<latent_185>,<latent_186>,<latent_187>,<latent_188>,<latent_189>,<latent_190>,<latent_191>,<latent_192>,<latent_193>,<latent_194>,<latent_195>,<latent_196>,<latent_197>,<latent_198>,<latent_199>,<latent_200>,<latent_201>,<latent_202>,<latent_203>,<latent_204>,<latent_205>,<latent_206>,<latent_207>,<latent_208>,<latent_209>,<latent_210>,<latent_211>,<latent_212>,<latent_213>,<latent_214>,<latent_215>,<latent_216>,<latent_217>,<latent_218>,<latent_219>,<latent_220>,<latent_221>,<latent_222>,<latent_223>,<latent_224>,<latent_225>,<latent_226>,<latent_227>,<latent_228>,<latent_229>,<latent_230>,<latent_231>,<latent_232>,<latent_233>,<latent_234>,<latent_235>,<latent_236>,<latent_237>,<latent_238>,<latent_239>,<latent_240>,<latent_241>,<latent_242>,<latent_243>,<latent_244>,<latent_245>,<latent_246>,<latent_247>,<latent_248>,<latent_249>,<latent_250>,<latent_251>,<latent_252>,<latent_253>,<latent_254>,<latent_255>,<latent_256>"

structure_weight=0.1
max_latent_count=4
max_latent_len=32


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
    --output_dir ../saves/model/AStarThought-V2-Qwen3.6-27B \
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
    --run_name AStarThought-V2-Qwen3.6-27B \
    --astar_thought_latent true \
    --structure_weight ${structure_weight} \
    --latent_begin_token "$latent_begin_token" \
    --latent_end_token "$latent_end_token" \
    --latent_tokens "$latent_tokens" \
    --add_tokens "$latent_begin_token,$latent_end_token,$latent_tokens" \
    --max_latent_count ${max_latent_count} \
    --max_latent_len ${max_latent_len} \
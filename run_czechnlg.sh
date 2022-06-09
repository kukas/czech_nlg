export HF_DATASETS_CACHE="/scratch/project/open-24-11/balharj/cache/datasets"
export TRANSFORMERS_CACHE="/scratch/project/open-24-11/balharj/cache"
# export WANDB_DISABLED="true"

    # --max_train_samples 10 \
    # --max_eval_samples 10 \
    # --max_predict_samples 10 \
python run_czechnlg.py \
    --do_train \
    --model_name_or_path Helsinki-NLP/opus-mt-en-cs \
    --dataset_name cs_restaurants \
    --output_dir /scratch/project/open-24-11/balharj/experiments_debug/opus-mt-en-cs-finetune-cs_restaurants \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --max_steps=20000 \
    --learning_rate 5e-6 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --overwrite_output_dir \
    --predict_with_generate \
    --report_to none

# python run_translation.py \
#     --do_train \
#     --model_name_or_path Helsinki-NLP/opus-mt-en-cs \
#     --dataset_name cs_restaurants \
#     --source_lang en_XX \
#     --target_lang cs_CZ \
#     --output_dir /scratch/project/open-24-11/balharj/experiments_debug/opus-mt-en-cs-finetune-cs_restaurants-50-samples \
#     --max_train_samples 10 \
#     --max_eval_samples 10 \
#     --max_predict_samples 10 \
#     --evaluation_strategy epoch \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --report_to none

# python run_czechnlg.py \
#     --model_name_or_path facebook/mbart-large-50-one-to-many-mmt \
#     --do_eval \
#     --do_predict \
#     --preprocessing_num_workers 32 \
#     --dataset_name cs_restaurants \
#     --output_dir /scratch/project/open-24-11/balharj/experiments/mbart-mmt-large-finetuned-cs_restaurants \
#     --per_device_train_batch_size=8 \
#     --gradient_accumulation_steps=4 \
#     --max_steps=20000 \
#     --metric_for_best_model=bleu \
#     --forced_bos_token cs_CZ \
#     --load_best_model_at_end \
#     --evaluation_strategy steps \
#     --fp16 \
#     --learning_rate 5e-6 \
#     --save_steps 100 \
#     --eval_steps 100 \
#     --per_device_eval_batch_size=8 \
#     --overwrite_output_dir \
#     --predict_with_generate

# python run_czechnlg.py \
#     --model_name_or_path facebook/mbart-large-cc25 \
#     --do_eval \
#     --do_train \
#     --preprocessing_num_workers 32 \
#     --dataset_name cs_restaurants \
#     --output_dir /scratch/project/open-24-11/balharj/experiments/mbart-large-cc25-finetuned-cs_restaurants \
#     --per_device_train_batch_size=8 \
#     --gradient_accumulation_steps=4 \
#     --max_steps=1000 \
#     --metric_for_best_model=bleu \
#     --forced_bos_token cs_CZ \
#     --load_best_model_at_end \
#     --evaluation_strategy steps \
#     --fp16 \
#     --learning_rate 5e-6 \
#     --save_steps 50 \
#     --eval_steps 50 \
#     --per_device_eval_batch_size=4 \
#     --overwrite_output_dir \
#     --predict_with_generate

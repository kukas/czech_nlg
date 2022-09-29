set -xe

export HF_DATASETS_CACHE="/scratch/project/open-26-22/balharj/cache/datasets"
export TRANSFORMERS_CACHE="/scratch/project/open-26-22/balharj/cache"

export WANDB_PROJECT=czechnlg_debug
# export WANDB_DISABLED="true"

# python run_czechnlg.py \
#     --do_train \
#     --do_eval \
#     --model_name_or_path Helsinki-NLP/opus-mt-en-cs \
#     --dataset_name cs_restaurants \
#     --output_dir /scratch/project/open-26-22/balharj/experiments_debug/opus-mt-en-cs-finetune-cs_restaurants \
#     --per_device_train_batch_size=8 \
#     --per_device_eval_batch_size=8 \
#     --max_steps=20000 \
#     --learning_rate 5e-6 \
#     --save_steps 100 \
#     --eval_steps 100 \
#     --evaluation_strategy steps \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --report_to none

# Testing
# python run_czechnlg.py \
#     --do_train \
#     --do_eval \
#     --model_name_or_path Helsinki-NLP/opus-mt-en-cs \
#     --dataset_name cs_restaurants \
#     --output_dir /scratch/project/open-26-22/balharj/experiments_debug/opus-mt-en-cs-finetune-cs_restaurants-10-samples \
#     --max_train_samples 100 \
#     --max_eval_samples 10 \
#     --max_predict_samples 10 \
#     --num_train_epochs 20 \
#     --evaluation_strategy epoch \
#     --save_strategy epoch \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --report_to none

# Testing mBART
# python run_czechnlg.py \
#     --do_train \
#     --do_eval \
#     --model_name_or_path facebook/mbart-large-50-one-to-many-mmt \
#     --dataset_name cs_restaurants \
#     --output_dir /scratch/project/open-26-22/balharj/experiments_debug/mbart-large-50-finetune-cs_restaurants-10-samples \
#     --max_train_samples 100 \
#     --max_eval_samples 10 \
#     --max_predict_samples 10 \
#     --num_train_epochs 20 \
#     --evaluation_strategy epoch \
#     --save_strategy epoch \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --report_to none

# Testing
# python run_czechnlg.py \
#     --do_train \
#     --do_eval \
#     --model_name_or_path facebook/m2m100_418M \
#     --dataset_name datasets/web_nlg/web_nlg.py \
#     --dataset_data_dir datasets/web_nlg/webnlg-dataset-cz \
#     --dataset_config_name release_v3.0_ru_cs \
#     --output_dir /scratch/project/open-26-22/balharj/experiments_debug/facebook/m2m100_418M-finetune-web_nlg \
#     --max_train_samples 1000 \
#     --max_eval_samples 100 \
#     --num_train_epochs 20 \
#     --evaluation_strategy steps \
#     --save_strategy steps \
#     --eval_steps 100 \
#     --save_steps 100 \
#     --predict_with_generate \
#     --report_to none

# Testing mT5
    # --per_device_train_batch_size=8 \
    # --gradient_accumulation_steps=4 \
    # --learning_rate 0.001 \

    # --max_train_samples 1000 \
    # --max_eval_samples 100 \

# python run_czechnlg.py \
#     --do_train \
#     --do_eval \
#     --model_name_or_path google/mt5-base \
#     --dataset_name cs_restaurants \
#     --output_dir /scratch/project/open-26-22/balharj/experiments_debug/mt5-base-finetune-cs_restaurants \
#     --num_train_epochs 20 \
#     --evaluation_strategy steps \
#     --save_strategy steps \
#     --eval_steps 100 \
#     --save_steps 100 \
#     --predict_with_generate \
#     --overwrite_output_dir \
#     --report_to none

    # --per_device_train_batch_size=8 \
    # --gradient_accumulation_steps=4 \
    # --forced_bos_token cs_CZ \
    # --fp16 \
    # --learning_rate 5e-6 \
    # --per_device_eval_batch_size=8 \

python run_czechnlg.py \
    --do_train \
    --do_eval \
    --model_name_or_path facebook/mbart-large-50-one-to-many-mmt \
    --dataset_name datasets/web_nlg/web_nlg.py \
    --dataset_data_dir datasets/web_nlg/webnlg-dataset-cz \
    --dataset_config_name release_v3.0_ru_cs \
    --output_dir /scratch/project/open-26-22/balharj/experiments_debug/mbart-large-mmt-finetuned-web_nlg \
    --num_train_epochs 20 \
    --max_train_samples 256 \
    --max_eval_samples 64 \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_steps 100 \
    --eval_steps 100 \
    --overwrite_output_dir \
    --predict_with_generate \
    --report_to wandb \
    --run_name mbart-large-mmt-finetuned-web_nlg

# python run_czechnlg.py \
#     --model_name_or_path facebook/mbart-large-cc25 \
#     --do_eval \
#     --do_train \
#     --preprocessing_num_workers 32 \
#     --dataset_name cs_restaurants \
#     --output_dir /scratch/project/open-26-22/balharj/experiments/mbart-large-cc25-finetuned-cs_restaurants \
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

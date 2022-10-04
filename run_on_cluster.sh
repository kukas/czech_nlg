set -xe

EXPERIMENTS_DIR="/scratch/project/open-26-22/balharj/experiments_new"
MODEL_NAME="facebook/m2m100_1.2B"
# replace slash with underscore
MODEL_NAME_PATH=${MODEL_NAME//\//_}

PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION=1

qsub -A OPEN-26-22 -q qnvidia -l select=1,walltime=2:00:00 -- /home/balharj/czech_nlg/run_czechnlg.sh \
    czechnlg_new \
    --do_train \
    --do_eval \
    --model_name_or_path $MODEL_NAME \
    --dataset_name datasets/web_nlg/web_nlg.py \
    --dataset_data_dir datasets/web_nlg/webnlg-dataset-cz \
    --dataset_config_name release_v3.0_ru_cs \
    --output_dir $EXPERIMENTS_DIR/$MODEL_NAME_PATH \
    --num_train_epochs 20 \
    --dataloader_num_workers 64 \
    --use_fast_tokenizer=False \
    --per_device_train_batch_size=$PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --per_device_eval_batch_size=4 \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --overwrite_output_dir \
    --predict_with_generate \
    --report_to wandb \
    --run_name $MODEL_NAME-finetuned-web_nlg


# python run_czechnlg.py \
#     --do_train \
#     --do_eval \
#     --model_name_or_path google/mt5-base \
#     --dataset_name datasets/web_nlg/web_nlg.py \
#     --dataset_data_dir datasets/web_nlg/webnlg-dataset-cz \
#     --dataset_config_name release_v3.0_ru_cs \
#     --output_dir /scratch/project/open-26-22/balharj/experiments/mt5-base-finetuned-web_nlg \
#     --skip_memory_metrics=False \
#     --num_train_epochs 20 \
#     --dataloader_num_workers 64 \
#     --per_device_train_batch_size=16 \
#     --gradient_accumulation_steps=2 \
#     --evaluation_strategy no \
#     --logging_strategy epoch \
#     --save_strategy epoch \
#     --overwrite_output_dir \
#     --report_to wandb \
#     --run_name mt5-base-finetuned-web_nlg



# OK
# python run_czechnlg.py \
#     --do_train \
#     --model_name_or_path facebook/m2m100_418M \
#     --dataset_name datasets/web_nlg/web_nlg.py \
#     --dataset_data_dir datasets/web_nlg/webnlg-dataset-cz \
#     --dataset_config_name release_v3.0_ru_cs \
#     --output_dir /scratch/project/open-26-22/balharj/experiments/m2m100_418M-finetuned-web_nlg \
#     --skip_memory_metrics=False \
#     --num_train_epochs 20 \
#     --dataloader_num_workers 64 \
#     --per_device_train_batch_size=16 \
#     --gradient_accumulation_steps=2 \
#     --evaluation_strategy no \
#     --logging_strategy epoch \
#     --save_strategy epoch \
#     --overwrite_output_dir \
#     --report_to wandb \
#     --run_name m2m100_418M-finetuned-web_nlg

# python run_czechnlg.py \
#     --do_train \
#     --model_name_or_path Helsinki-NLP/opus-mt-en-cs \
#     --dataset_name datasets/web_nlg/web_nlg.py \
#     --dataset_data_dir datasets/web_nlg/webnlg-dataset-cz \
#     --dataset_config_name release_v3.0_ru_cs \
#     --output_dir /scratch/project/open-26-22/balharj/experiments/opus-mt-finetuned-web_nlg \
#     --skip_memory_metrics=False \
#     --num_train_epochs 20 \
#     --dataloader_num_workers 64 \
#     --per_device_train_batch_size=16 \
#     --gradient_accumulation_steps=2 \
#     --evaluation_strategy no \
#     --logging_strategy epoch \
#     --save_strategy epoch \
#     --overwrite_output_dir \
#     --report_to wandb \
#     --run_name opus-mt-finetuned-web_nlg

# python run_czechnlg.py \
#     --do_train \
#     --do_eval \
#     --model_name_or_path google/mt5-base \
#     --dataset_name datasets/web_nlg/web_nlg.py \
#     --dataset_data_dir datasets/web_nlg/webnlg-dataset-cz \
#     --dataset_config_name release_v3.0_ru_cs \
#     --output_dir /scratch/project/open-26-22/balharj/experiments/mt5-base-finetuned-web_nlg \
#     --skip_memory_metrics=False \
#     --num_train_epochs 20 \
#     --dataloader_num_workers 64 \
#     --per_device_train_batch_size=16 \
#     --gradient_accumulation_steps=2 \
#     --evaluation_strategy no \
#     --logging_strategy epoch \
#     --save_strategy epoch \
#     --overwrite_output_dir \
#     --report_to wandb \
#     --run_name mt5-base-finetuned-web_nlg


set -xe

MODEL_NAME="facebook/nllb-200-distilled-600M"
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION=1

./run_czechnlg.sh \
    czechnlg_debug \
    --do_train \
    --model_name_or_path $MODEL_NAME \
    --dataset_name datasets/web_nlg/web_nlg.py \
    --dataset_data_dir datasets/web_nlg/webnlg-dataset-cz \
    --dataset_config_name release_v3.0_ru_cs \
    --output_dir /scratch/project/open-26-22/balharj/experiments_debug/$MODEL_NAME \
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


export HF_DATASETS_CACHE="/scratch/project/open-24-11/balharj/cache/datasets"
export TRANSFORMERS_CACHE="/scratch/project/open-24-11/balharj/cache"
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -m pdb -c continue run_translation.py \
    --model_name_or_path Helsinki-NLP/opus-mt-en-cs \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang cs \
    --dataset_name wmt19 \
    --dataset_config_name cs-en \
    --output_dir /scratch/project/open-24-11/balharj/experiments/opus-mt-en-cs-finetuned-wmt19-single \
    --per_device_train_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --max_steps=100000 \
    --metric_for_best_model=bleu \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --fp16 \
    --save_steps 500 \
    --eval_steps 500 \
    --per_device_eval_batch_size=8 \
    --overwrite_output_dir \
    --predict_with_generate
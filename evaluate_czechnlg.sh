set -xe

export HF_DATASETS_CACHE="/scratch/project/open-26-22/balharj/cache/datasets"
export TRANSFORMERS_CACHE="/scratch/project/open-26-22/balharj/cache"
export HF_METRICS_CACHE="/scratch/project/open-26-22/balharj/cache/metrics"

export WANDB_PROJECT=czechnlg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

python compute_heavy_metrics.py --outputs_csv /scratch/project/open-26-22/balharj/experiments_new/facebook/m2m100_418M/outputs_eval-1145.csv --split validation --small_models

# OK
# python run_czechnlg.py \
#     --do_eval \
#     --model_name_or_path facebook/mbart-large-cc25 \
#     --dataset_name datasets/web_nlg/web_nlg.py \
#     --dataset_data_dir datasets/web_nlg/webnlg-dataset-cz \
#     --dataset_config_name release_v3.0_ru_cs \
#     --output_dir /scratch/project/open-26-22/balharj/experiments/mbart-large-cc25-finetuned-web_nlg \
#     --dataloader_num_workers 0 \
#     --max_eval_samples 64 \
#     --report_to none \
#     --predict_with_generate \
#     --run_name mbart-large-cc25-finetuned-web_nlg

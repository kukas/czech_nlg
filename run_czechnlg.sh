#!/bin/bash
source ~/.bashrc
set -xe

export WANDB_PROJECT=$1

export HF_DATASETS_CACHE="/scratch/project/open-26-22/balharj/cache/datasets"
export TRANSFORMERS_CACHE="/scratch/project/open-26-22/balharj/cache"
export HF_METRICS_CACHE="/scratch/project/open-26-22/balharj/cache/metrics"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

# change current working directory to the script location
cd /home/balharj/czech_nlg

# activate conda for subshell execution
CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate nlg

python run_czechnlg.py "${@:2}"
#!/bin/bash
#source ~/.bashrc

# change current working directory to the script location
cd /home/balharj/czech_nlg

# activate conda for subshell execution
CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
# source /home/balharj/miniconda3/etc/profile.d/conda.sh

conda activate nlg
export HF_DATASETS_CACHE="/scratch/project/open-24-11/balharj/cache/datasets"
export TRANSFORMERS_CACHE="/scratch/project/open-24-11/balharj/cache"
python train_text2text.py --do-train --do-eval --do-predict --epochs 714 --learning-rate 1e-4 --weight-decay 0 --logging-strategy steps --scratch-dir /scratch/project/open-24-11/balharj/experiments

# python "$@"
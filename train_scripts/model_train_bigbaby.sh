#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=30:00:00   # Specify the hard time limit for the job
#$ -N bigbaby  # Project name.  unique every time 
#$ -o std_out_bigbaby # standard out file
#$ -e err_bigbaby # error file
#$ -l gpus=1
#$ -l gpu_memory=60G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "python train.py LlamaBaby fetus_config_large_1.json 20 CPUDoubleFileDataset negativeten_ten big_baby_10 --lr 0.0001"
echo "model_train_bigbaby.sh"
cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

# run job
# for later, make it so you can specify save file name, and get rid of logs, scc handles logs
cd /projectnb/textconv/llama
python train.py LlamaBaby fetus_config_large_1.json 100 CPUDoubleFileDataset negativeten_ten big_baby_10 --lr 0.000001 --resume --clip
# with 20 files this takes 5 minutes with 1 worker.  
# with 4 workers: 20 files takes 
# turned up the lr, resubmitting for 30 hours  
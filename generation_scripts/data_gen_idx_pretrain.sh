#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=04:00:00   # Specify the hard time limit for the job
#$ -N dgidxpre  # Project name.  unique every time 
#$ -o std_out_data_gen_idx_pretrain # standard out file
#$ -e err_data_gen_idx_pretrain # error file
#$ -l gpu_c=8.6
#$ -l gpus=1
#$ -pe omp 2
#$ -V
#$ -m e

# 100 100 takes half an hour
# doing 1000 100, requested 7 hours to be safe. 
# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 

cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh

# run job
cd /projectnb/textconv/llama
python llama_idx_data_gen.py 100 100 idx_pretrain

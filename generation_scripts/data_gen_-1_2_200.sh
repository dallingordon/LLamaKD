#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=05:00:00   # Specify the hard time limit for the job
#$ -N gen_-10full  # Project name.  unique every time 
#$ -o std_out_-10full # standard out file
#$ -e err_gen-10full # error file
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
python llama_data_gen.py 200 100 -1 2 expanded_range_200_full 1.0 #added the arguments, this has been ran, is in llama_data/

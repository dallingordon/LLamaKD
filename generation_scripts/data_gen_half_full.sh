#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=07:00:00   # Specify the hard time limit for the job
#$ -N datagen2  # Project name.  unique every time 
#$ -o std_out_datagen2 # standard out file
#$ -e err_datagen2 # error file
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
python llama_data_gen.py 1000 100 -1 2 half_full 0.5 #added the arguments, this has been ran, is in llama_data/

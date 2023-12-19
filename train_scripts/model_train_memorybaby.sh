#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=20:00:00   # Specify the hard time limit for the job
#$ -N membaby  # Project name.  unique every time 
#$ -o std_out_membaby # standard out file
#$ -e err_membaby # error file
#$ -l gpu_c=7.0
#$ -l gpu_memory=40G
#$ -l gpus=1
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_memorybaby.sh"
echo "python train.py MemoryBaby mem_config_1.json 20 CPUDoubleFileDataset negativeten_ten memory_baby_1 --lr 0.0001"
cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh

# run job
# for later, make it so you can specify save file name, and get rid of logs, scc handles logs
cd /projectnb/textconv/llama
python train.py MemoryBaby mem_config_1.json 50 CPUDoubleFileDataset negativeten_ten memory_baby_1 --lr 0.000001 --clip --resume

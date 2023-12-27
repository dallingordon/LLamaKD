#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=40:00:00   # Specify the hard time limit for the job
#$ -N memCbbpr  # Project name.  unique every time 
#$ -o std_out_SMCrossBaby_pre # standard out file
#$ -e err_SMCrossBaby_pre # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_SMCrossBaby_pre.sh"
echo "python train.py SMCrossBaby_1 SMCrossBaby_1_mem500.json 20 IdxDataset idx_pretrain SMCrossBaby_1_pre --lr 0.00001 --clip --resume
python train.py SMCrossBaby_1 SMCrossBaby_1_mem500.json 50 CPUDoubleFileDataset negativeten_ten SMCrossBaby_1_pre --lr 0.00001 --clip --resume
python train.py SMCrossBaby_1 SMCrossBaby_1_mem500.json 5 IdxDataset idx_pretrain SMCrossBaby_1_pre --lr 0.00001 --clip --resume"

cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

cd /projectnb/textconv/llama
python train.py SMCrossBaby_1 SMCrossBaby_1_mem500.json 20 IdxDataset idx_pretrain SMCrossBaby_1_pre --lr 0.00001 --clip --resume
python train.py SMCrossBaby_1 SMCrossBaby_1_mem500.json 50 CPUDoubleFileDataset negativeten_ten SMCrossBaby_1_pre --lr 0.00001 --clip --resume
python train.py SMCrossBaby_1 SMCrossBaby_1_mem500.json 5 IdxDataset idx_pretrain SMCrossBaby_1_pre --lr 0.00001 --clip --resume

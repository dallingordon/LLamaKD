#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=40:00:00   # Specify the hard time limit for the job
#$ -N mCCbbpre  # Project name.  unique every time 
#$ -o std_out_SMCrossBaby_Concat_pre # standard out file
#$ -e err_SMCrossBaby_Concat_pre # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_SMCrossBaby_Concat_pre.sh"
echo "python train.py SMCrossBaby_Concat SMCrossBaby_concat1k.json 40 IdxDataset idx_pretrain SMCrossBaby_C1k_pre --lr 0.000001 --clip --resume
python train.py SMCrossBaby_Concat SMCrossBaby_concat1k.json 40 CPUDoubleFileDataset negativeten_ten SMCrossBaby_C1k_pre --lr 0.000001 --clip --resume"

cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

cd /projectnb/textconv/llama

python train.py SMCrossBaby_Concat SMCrossBaby_concat1k.json 40 IdxDataset idx_pretrain SMCrossBaby_C1k_pre --lr 0.000001 --clip --resume
python train.py SMCrossBaby_Concat SMCrossBaby_concat1k.json 40 CPUDoubleFileDataset negativeten_ten SMCrossBaby_C1k_pre --lr 0.000001 --clip --resume

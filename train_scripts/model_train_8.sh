#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=40:00:00   # Specify the hard time limit for the job
#$ -N m_8  # Project name.  unique every time 
#$ -o std_out_m_8 # standard out file
#$ -e err_m_8 # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_8.sh"


cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

cd /projectnb/textconv/llama
# python train.py Model_2 Model_x_config.json 10 OHIdxDataset idx_pretrain m_8 --lr 0.00001 --clip
python train.py Model_2 Model_x_config.json 20 IdxDataset idx_pretrain m_8 --lr 0.001 --clip --he
python train.py Model_2 Model_x_config.json 50 CPUDoubleFileDataset negativeten_ten m_8 --lr 0.0001 --clip --resume

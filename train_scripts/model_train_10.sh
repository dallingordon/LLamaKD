#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=40:00:00   # Specify the hard time limit for the job
#$ -N m_10  # Project name.  unique every time 
#$ -o std_out_m_10 # standard out file
#$ -e err_m_10 # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_10.sh"


cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

cd /projectnb/textconv/llama
python train.py Model_4 Model_x_config.json 10 OHIdxDataset idx_pretrain m_10 --lr 0.00001 --clip
python train.py Model_4 Model_x_config.json 20 IdxDataset idx_pretrain m_10 --lr 0.000001 --clip --resume
python train.py Model_4 Model_x_config.json 50 CPUDoubleFileDataset negativeten_ten m_10 --lr 0.0000001 --clip --resume

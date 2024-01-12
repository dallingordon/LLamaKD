#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=40:00:00   # Specify the hard time limit for the job
#$ -N m_7  # Project name.  unique every time 
#$ -o std_out_m_7 # standard out file
#$ -e err_m_7 # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_7.sh"


cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

cd /projectnb/textconv/llama
# python train.py Model_1 Model_x_config.json 10 OHIdxDataset idx_pretrain m_7 --lr 0.00001 --clip
python train.py Model_1 Model_x_config.json 15 IdxDataset idx_pretrain m_7 --lr 0.001 --clip --he --CE --ceweight 10.0
python train.py Model_1 Model_x_config.json 50 CPUDoubleFileDataset negativeten_ten m_7 --lr 0.00001 --clip --resume --CE --ceweight 5.0

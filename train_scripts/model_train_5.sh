#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=40:00:00   # Specify the hard time limit for the job
#$ -N m_5  # Project name.  unique every time 
#$ -o std_out_m_5 # standard out file
#$ -e err_m_5 # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_LearnedDoubleCrossBabyWithSinPosEmb_pre.sh"
echo ""

cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

cd /projectnb/textconv/llama

python train.py LearnedDoubleCrossBabyWithSinEmbedding LearnedDoubleCrossBabyWithSinEmbedding_conf.json 50 CPUDoubleFileDataset negativeten_ten m_5 --lr 0.0001 --clip --he --CE  --ceweight 10.0
python train.py LearnedDoubleCrossBabyWithSinEmbedding LearnedDoubleCrossBabyWithSinEmbedding_conf.json 10 IdxDataset idx_pretrain m_5 --lr 0.0000001 --clip --resume
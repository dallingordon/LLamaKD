#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=40:00:00   # Specify the hard time limit for the job
#$ -N m_3  # Project name.  unique every time 
#$ -o std_out_m_3 # standard out file
#$ -e err_m_3 # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_3.sh"
echo "python train.py LearnedDoubleCrossBabyWithLearnedSinEmbedding LearnedDoubleCrossBabyWithLearnedSinEmbedding_conf.json 50 CPUDoubleFileDataset negativeten_ten m_3 --lr 0.0001 --he
python train.py LearnedDoubleCrossBabyWithLearnedSinEmbedding LearnedDoubleCrossBabyWithLearnedSinEmbedding_conf.json 10 IdxDataset idx_pretrain m_3 --lr 0.000001 --resume"

cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

cd /projectnb/textconv/llama
python train.py LearnedDoubleCrossBabyWithLearnedSinEmbedding LearnedDoubleCrossBabyWithLearnedSinEmbedding_conf.json 30 CPUDoubleFileDataset negativeten_ten m_3 --lr 0.0001 --he --clip --CE --ceweight 2.0
python train.py LearnedDoubleCrossBabyWithLearnedSinEmbedding LearnedDoubleCrossBabyWithLearnedSinEmbedding_conf.json 10 IdxDataset idx_pretrain m_3 --lr 0.000001 --resume --clip --CE
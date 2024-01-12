#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=40:00:00   # Specify the hard time limit for the job
#$ -N m_4  # Project name.  unique every time 
#$ -o std_out_m_4 # standard out file
#$ -e err_m_4 # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_4.sh"
echo "python train.py LearnedDoubleCrossBabyWithBinaryEmbedding LearnedDoubleCrossBaby_withbinaryemb_conf.json 100 CPUDoubleFileDataset negativeten_ten m_4 --lr 0.00001 --he
python train.py LearnedDoubleCrossBabyWithBinaryEmbedding LearnedDoubleCrossBaby_withbinaryemb_conf.json 10 IdxDataset idx_pretrain m_4 --lr 0.00001 --resume --clip"

cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

cd /projectnb/textconv/llama
#python train.py LearnedDoubleCrossBabyWithBinaryEmbedding LearnedDoubleCrossBaby_withbinaryemb_conf.json 20 IdxDataset idx_pretrain m_4 --lr 0.00001 --clip --resume
python train.py LearnedDoubleCrossBabyWithBinaryEmbedding LearnedDoubleCrossBaby_withbinaryemb_conf.json 20 CPUDoubleFileDataset negativeten_ten m_4 --lr 0.0001 --he --clip --CE --ceweight 2.0
python train.py LearnedDoubleCrossBabyWithBinaryEmbedding LearnedDoubleCrossBaby_withbinaryemb_conf.json 10 IdxDataset idx_pretrain m_4 --lr 0.000001 --resume --clip --CE  --ceweight 5.0
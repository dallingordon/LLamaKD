#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=40:00:00   # Specify the hard time limit for the job
#$ -N m_6  # Project name.  unique every time 
#$ -o std_out_m_6 # standard out file
#$ -e err_m_6 # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_6.sh"
echo ""

cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

cd /projectnb/textconv/llama

python train.py LearnedDoubleCrossBabyWithLearnedSinEmbedding LearnedDoubleCrossBabyWithLearnedSinEmbedding_conf.json 50 CPUDoubleFileDataset negativeten_ten m_6 --lr 0.0001 --clip --he --CE  --ceweight 2.0

#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=40:00:00   # Specify the hard time limit for the job
#$ -N LDCBWPE  # Project name.  unique every time 
#$ -o std_out_LearnedDoubleCrossBabyWithPosEmb_pre # standard out file
#$ -e err_LearnedDoubleCrossBabyWithPosEmb_pre # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_LearnedDoubleCrossBabyWithPosEmb_pre.sh"
echo ""

cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

cd /projectnb/textconv/llama
python train.py LearnedDoubleCrossBabyWithBinaryEmbedding LearnedDoubleCrossBaby_withbinaryemb_conf.json 20 IdxDataset idx_pretrain LDCBWPE --lr 0.00001 --clip 
python train.py LearnedDoubleCrossBabyWithBinaryEmbedding LearnedDoubleCrossBaby_withbinaryemb_conf.json 50 CPUDoubleFileDataset negativeten_ten LDCBWPE --lr 0.000001 --clip --resume
python train.py LearnedDoubleCrossBabyWithBinaryEmbedding LearnedDoubleCrossBaby_withbinaryemb_conf.json 10 IdxDataset idx_pretrain LDCBWPE --lr 0.00001 --clip --resume
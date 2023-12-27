#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=30:00:00   # Specify the hard time limit for the job
#$ -N Cbb2pre  # Project name.  unique every time 
#$ -o std_out_CrossBaby_2_pre # standard out file
#$ -e err_CrossBaby_2_pre # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_CrossBaby_2_pre.sh"
echo "python train.py CrossBaby_2 CrossBab1_2_50.json 20 IdxDataset idx_pretrain CrossBB_2_pretrain --lr 0.001 --clip 
python train.py CrossBaby_2 CrossBaby_2_50.json 100 CPUDoubleFileDataset negativeten_ten CrossBB_2_pretrain --lr 0.000001 --clip --resume"

cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh 

cd /projectnb/textconv/llama
python train.py CrossBaby_2 CrossBaby_2_50.json 20 IdxDataset idx_pretrain CrossBB_2_pretrain --lr 0.001 --clip 
python train.py CrossBaby_2 CrossBaby_2_50.json 100 CPUDoubleFileDataset negativeten_ten CrossBB_2_pretrain --lr 0.000001 --clip --resume

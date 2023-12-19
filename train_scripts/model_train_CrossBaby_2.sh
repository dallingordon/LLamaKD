#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=30:00:00   # Specify the hard time limit for the job
#$ -N crossbb2  # Project name.  unique every time 
#$ -o std_out_CrossBaby_2 # standard out file
#$ -e err_CrossBaby_2 # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_CrossBaby_2.sh"
echo "python train.py CrossBaby_2 CrossBaby_2_50.json 50 CPUDoubleFileDataset negativeten_ten CrossBaby_2_50 --lr 0.000001 --clip "

cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

cd /projectnb/textconv/llama
python train.py CrossBaby_2 CrossBaby_2_50.json 50 CPUDoubleFileDataset negativeten_ten CrossBaby_2_50 --lr 0.000001 --clip

#!/bin/bash -l

echo "running script"
#$ -P textconv       # Specify the SCC project name you want to use
#$ -l h_rt=30:00:00   # Specify the hard time limit for the job
#$ -N memCbb  # Project name.  unique every time 
#$ -o std_out_SMCrossBaby_1 # standard out file
#$ -e err_SMCrossBaby_1 # error file
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -pe omp 4
#$ -V
#$ -m e

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 
echo "model_train_SMCrossBaby_1_500.sh"
echo "python train.py SMCrossBaby_1 SMCrossBaby_1_mem500.json 200 CPUDoubleFileDataset negativeten_ten SMCrossBaby_1_500 --lr 0.00001 --clip"

cd /projectnb/textconv/llama/scc_scripts

source scc_setup.sh # not sure

cd /projectnb/textconv/llama
python train.py SMCrossBaby_1 SMCrossBaby_1_mem500.json 200 CPUDoubleFileDataset negativeten_ten SMCrossBaby_1_500 --lr 0.00001 --clip

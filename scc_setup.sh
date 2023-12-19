#!/bin/bash

echo "Loading Modules LLAMA"
module load python3/3.10.5
# pip install --no-cache-dir --prefix=/projectnb/textconv/llama/packages torch==2.0.1
module load pytorch/1.13.1
module load gcc/8.3.0
module load cuda/11.1
# added these comments on 11/5, not sure i need to install each time.  try this on a qrsh next time.
# echo "Installing llama"
cd /projectnb/textconv/llama
# pip install --target=/projectnb/textconv/llama/packages -r requirements.txt
# pip install -e . --prefix=/projectnb/textconv/llama/packages
export PYTHONPATH="/projectnb/textconv/llama/packages"
echo $PYTHONPATH
echo "all done"


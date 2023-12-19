#!/usr/bin/env python
# coding: utf-8


#example usage python llama_data_gen.py 100 100 -1 2 folder 0.5
# the first 100 is the number of files (see line 37)
# the second is the batch dim of the tensor of those files, samples

import sys

sys.path.append('/projectnb/textconv/llama/packages')

import fairscale





from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
import torch
import random





from llama.generation import Llama, Dialog
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer



n_files = int(sys.argv[1])
files_per_n = int(sys.argv[2])
alpha = float(sys.argv[3])
beta = float(sys.argv[4])
dest_folder = str(sys.argv[5]).strip()
full_percent = float(sys.argv[6]) #this is how often it does a full max_len tensor of random inputs

# Check if CUDA is available rip sam altman (as ceo, i hope he doesn't die)
if torch.cuda.is_available():
    # Get the number of CUDA devices
    num_cuda_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_cuda_devices}")

    # List the properties of each CUDA device
    for i in range(num_cuda_devices):
        device = torch.device(f'cuda:{i}')
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available on this system.")





import os
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '49154' #49152-65535 will need to be dynamic.  some script maybe?
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'





generator = Llama.build(
        ckpt_dir="llama-2-7b-chat/",
        tokenizer_path="tokenizer.model",
        max_seq_len=512, #max_seq_len....
        max_batch_size=6,
    )





#from NoiseKD
import torch.nn as nn
def Linearize_Embedding(embedding_layer):
    embedding_weight_tensor = embedding_layer.weight.detach() 
    shape = embedding_weight_tensor.shape
    vocab_size = shape[0]
    embedding_dim = shape[1]
    lin = nn.Linear(vocab_size,embedding_dim, bias = False)
    #print(lin.weight.shape)
    #print(embedding_weight_tensor.shape)
    lin.weight = nn.Parameter(embedding_weight_tensor.T) #not sure about this transpose
    return lin

def batch_one_hot(input_sequences, vocab_size):
    batch_size = input_sequences.size(0)
    max_seq_length = input_sequences.size(1)
    
    # Create a tensor to store the one-hot encodings
    one_hot_input = torch.zeros(batch_size, max_seq_length, vocab_size)
    
    # Use scatter_ to set the appropriate elements to 1 in each batch
    one_hot_input.scatter_(2, input_sequences.unsqueeze(2), 1)
    return one_hot_input





float_embeddings = Linearize_Embedding(generator.model.tok_embeddings)


# In[11]: # i definitely didn't make this in a jupyter notebook.


generator.model.tok_embeddings = float_embeddings #set, now it takes one hots




def random_float_tensor(a = 0.0 
                        ,b = 1.0
                        ,max_percent = 0.5 #how often to do a full input tensor
                        ,max_len=512
                       ,vocab_size =32_000 ):
        # Replace with your desired lower bound
         # Replace with your desired upper bound

    #random_int = random.randint(1, max_len)  #this is the random_input lenght
    random_int = 0
    if random.random() < max_percent:
        random_int = max_len
    else:
        random_int = random.randint(1, max_len - 1)
    # Generate the random tensor
    random_tensor = torch.FloatTensor(1, random_int,vocab_size ).uniform_(a, b).to(torch.float16).to("cpu")
    #print(random_tensor.device)
    zero_tensor = torch.zeros(1, max_len - random_int, vocab_size, dtype=torch.float16).to("cpu") #this stays on the cpu for saving.
    #print(zero_tensor.device)
    cpu_tensor = torch.cat((random_tensor, zero_tensor), dim=1)
    
    return random_tensor,cpu_tensor 




#data_dir =  "/projectnb/textconv/llama/llama_data/" #get this from an arg i think
data_dir =  f"/projectnb/textconv/llama/generated_datasets/{dest_folder}/"
if not os.path.exists(data_dir):
    # If it doesn't exist, create it
    os.makedirs(data_dir)


files_n = n_files #how many files to generate
batch_per_file = files_per_n #how many batches in each file.
vocab_size = 32_000
max_len = 512
#a = -1.0
#b = 2.0

#
a = alpha 
b = beta 


#find files, for resuming mostly
max_number = -1
for filename in os.listdir(data_dir):
    if filename.startswith("input") and filename.endswith(".pt"):
        try:
            number = int(filename[len("input"):-len(".pt")])
            max_number = max(max_number, number)
        except ValueError:
            pass

if max_number != -1:
    print(f"The maximum file number found is: {max_number}")
else:
    print("No matching files found in the directory.")


#for i in range(files_n):
for i in range(max_number + 1, max_number + files_n + 1):
    input_tensor = torch.zeros(batch_per_file
                               ,max_len
                               ,vocab_size
                               ,dtype=torch.float16)
    target_tensor = torch.zeros(batch_per_file
                                ,vocab_size
                               ,dtype=torch.float32)
    for j in range(batch_per_file):
        pred_tensor,save_tensor = random_float_tensor(a
                                                      ,b
                                                      ,full_percent
                                                      ,max_len=512
                       ,vocab_size =32_000)
        input_tensor[j] = save_tensor
        pred_tensor = pred_tensor.to(device)
        model_output = generator.model.forward(pred_tensor,0)
        target_tensor[j] = model_output[:,-1,:] #need to validate this
    #print(f"input{i}.pt")
    #print(input_tensor.shape,target_tensor.shape)
    #print(f"target{i}.pt")
    ##save
    input_path = f"{data_dir}input{i}.pt"
    target_path = f"{data_dir}target{i}.pt"
    torch.save(input_tensor, input_path)
    torch.save(target_tensor, target_path)





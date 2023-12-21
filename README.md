# LLamaKD
Llama Knowledge Distillation Research

Fall 2023 research project. 
LLMs typically have as their firs layer an Embedding layer.  
These are optimized to take an integer index as input. 
The weight matrices for this layer is vocab_size x embedding_dim.  
With small alterations these layers can take floats as inputs.  
This project attempts to distill LLM knowlege from a pre-trained LLM (llama 7B) by performing the Embedding alteration (which does not change pre-trained weights), passing noise through the LLM, then storing the resultant logits.  After Distilling the student model has the opposite operation performed on its first layer so its input is integer indexes and its performance can be directly compared to the LLM teacher.  
# Data Generation

llama_data_gen.py generates the training data in the form of tensors stored as .pt files.  The code to alter the Teacher LLM's embedding is stored here. 
It is ran from the command line.  An example script:

```python llama_data_gen.py 200 100 -10 10 negativeten_ten 0.5```

The first integer is the number of files to generate, the second is the number of samples, corresponding to the batch dimension in the resultant .pt files.
The next two integers are the range for the random noise inputs, in this case floats between -10 and 10. 
'negativeten_ten' is the name of the dataset, it will make a folder under /generated_datasets by this name, and store the input and output tensors there.  
This is specified in the train call.  The final float is between 0 and 1 and is the percentage of the input data that should be the full sequence length.  in this case, half will be full, the rest will be random lengths less than the max_seq_len. 

Pre-training on random indexes is accomplished with llama_idx_data_gen.py:

```python llama_idx_data_gen.py 200 100 idx_pretrain```

These generate full input length sequences of random token indexes.  Since these are stored as indexes, a separate dataloader that turns these into one-hot float tensors is used.    

Data generating .sh files are in /generation_scripts.
# Training
train.py Trains models.  It is executed from the command line:
```python train.py CrossBaby_1 CrossBab1_1_50.json 50 CPUDoubleFileDataset negativeten_ten CrossBaby_1_50 --lr 0.000001 --clip --resume```
The first string is the name of the model class to use.  The model classes are defined in student_models.py.
the second string is passed as **kwargs to the model.  These of course are model specific.  These .json files are in /configs.
'50' is the number of epochs to train.  
CPUDoubleFileDataset is the dataset/dataloader to use.  This is a class in kd_data_sets.py.
'negativeten_ten' is the dataset to use.  This was specified when the data was generated.  
'CrossBaby_1_50' is a save name.  The model is stored with this name.  
The learning rate can be specified.  Adding --clip at the end will clip gradients to 1.0, and --resume will load the model based on the specified save name.  

# Other Files

model_develop_config_generator.ipynb is used to count parameters and generate .json configs. 
_llama_kd_eval_notebook.ipynb performs the necessary operations on the Student models to revert back to an index accepting embedding layer.  This uses the provided Llama tokenizer as well as input to compare to the students.  



import argparse
import student_models #put models here. 
import kd_data_sets #
import torch
import json
import sys
import os
import time
from torch.utils.data import DataLoader
import torch.nn as nn


## _______________ model______model config______epochs_dataset in kdf data      savename
## python train.py LlamaFetus small_config.json 100 DoubleFileDataset half_full fetus_half_full
## python train.py MemoryLlama mem_config_1.json 3 CPUDoubleFileDataset half_full mem_debug1
## python train.py MemoryLlama mem_config_1.json 2 CPUDoubleFileDataset folder testbb --lr 0.0001
## python train.py CrossBaby_1 CrossBab1_1_50.json 5 IdxDataset idx_pretrain CrossBB_pretrain --lr 0.00001 --clip 

def train_model(modelname, kwargsfile, epochs, datasetname,datafolder, savename, lr, resume, clip, he, CE, ceweight):
    # Load the kwargs from the specified file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} will be used to train", file=sys.stdout)
    
    with open(kwargsfile, 'r') as f:
        kwargs = json.load(f)
    print("model kwargs loaded:", file=sys.stdout)
    print(kwargs, file=sys.stdout)
    # Instantiate the selected model using the kwargs
    if hasattr(student_models, modelname):
        model_class = getattr(student_models, modelname)
        pre_model_load_memory = torch.cuda.memory_allocated()
        model = model_class(**kwargs)
        model = model.to(device)
        post_model_load_memory = torch.cuda.memory_allocated()
        print(f"Model memory usage: {(post_model_load_memory - pre_model_load_memory) / (1024 ** 2):.2f} MB", file=sys.stdout)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters in the model: {total_params}", file=sys.stdout)
    else:
        print(f"Model '{modelname}' not found in student_models module.", file=sys.stdout)
        print("available modules", dir(student_models), file=sys.stdout)
        sys.exit(1)
        
    if hasattr(kd_data_sets, datasetname):
        dataset_class = getattr(kd_data_sets, datasetname)
        #data_dir = "/projectnb/textconv/llama/llama_data/"  #fix as in make dynamic.
        data_dir =  f"/projectnb/textconv/llama/generated_datasets/{datafolder}/"
        pre_data_mem = torch.cuda.memory_allocated()
        dataset = dataset_class(data_dir)  #pass these in later. 
        post_data_mem = torch.cuda.memory_allocated()
        dataset_usage = post_data_mem - pre_data_mem
        if dataset_usage > 1024:
            print(f"Model memory usage: {dataset_usage / (1024 ** 3):.2f} GB", file=sys.stdout)
        else:
            print(f"Model memory usage: {dataset_usage / (1024 ** 2):.2f} MB", file=sys.stdout)
        #running out of memory.  lazy load!!
        
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True,num_workers=4,prefetch_factor=4) #also here this is speedy
    else:
        print(f"Dataloader '{datasetname}' not found in kd_data_loaders module.", file=sys.stdout)
        sys.exit(1)

    # Print model architecture and kwargs to stdout
    print(f"Model Architecture:\n{model}\n", file=sys.stdout)
    print(f"Model Hyperparameters:\n{json.dumps(kwargs, indent=4)}\n", file=sys.stdout)
    print(f"Training for {epochs} epochs...\n", file=sys.stdout)

    # Your training loop here
    criterion = torch.nn.MSELoss() #i think this is the right call? idk yet.  
    criterion_ce = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 
    print(f"optimizer set with lr={lr}", file=sys.stdout)
    os.makedirs('models', exist_ok=True)
    if clip:
        print("gradient clipping turned on", file=sys.stdout)
    if resume:
        #load model
        weights_path = os.path.join('models', f'{savename}_trained.pth')
        if os.path.exists(weights_path):
            # Load model
            model.load_state_dict(torch.load(weights_path))
            print(f"Loaded from '{weights_path}'", file=sys.stdout)
        else:
            # File does not exist
            print(f"File '{weights_path}' not found", file=sys.stdout)
    
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Trainable: {param.requires_grad}", file=sys.stdout)
    
    
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            #print(m.weight)
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #print("after init:",m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        print("he init done!")
    
    if he:
        model.apply(init_weights)
        #he shook things up, experimenting with it for now.  
    
    model.train()
    
    for epoch in range(epochs):
        
        start_time = time.time()
        distinct_classes = set()
        
        for data, labels in dataloader:
            optimizer.zero_grad()
            data = data.to(device,dtype=torch.float32)
            labels = labels.to(device,dtype=torch.float32) #not sure this is the right place to put onto device.  maybe do it in the dataset? idk
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            output_argmax = torch.argmax(outputs,dim=-1)
            for i in output_argmax:
                distinct_classes.add(i.item())
            
            if CE:
                label_indices = torch.argmax(labels,dim=-1).long()
                loss_ce = criterion_ce(outputs, label_indices)
                
                # Combine losses
                combined_loss = loss + ceweight*loss_ce #ceweight is optional now.

                # Backward pass
                combined_loss.backward()
            else:
                loss.backward()
                
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            
            
        mse = loss.item()  # This assumes that your loss is already MSE
        total_classes = len(distinct_classes)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Epoch took {hours:02d}:{minutes:02d}:{seconds:02d}", file=sys.stdout)
        print(f"Epoch {epoch+1}/{epochs}, mse: {mse:.4f}, total classes predicted: {total_classes}", file=sys.stdout)
        
        if CE:
            cel = loss_ce.item()
            print(f"\tCross Entropy Loss: {cel:.4f}")
        
        model_save_path = os.path.join('models', f'{savename}_trained.pth')
        torch.save(model.state_dict(), model_save_path)
    #i want to save every so many epochs too.  
    model.eval()
    model_save_path = os.path.join('models', f'{savename}_trained.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to: {model_save_path}\n", file=sys.stdout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch student model")
    parser.add_argument("modelname", type=str, help="Name of the model class in student_models")
    parser.add_argument("kwargsfile", type=str, help="Path to the JSON file containing model kwargs")
    parser.add_argument("epochs", type=int, help="Number of training epochs")
    #parser.add_argument("stdoutfile", type=str, help="Path to the standard output log file")
    #parser.add_argument("stderrfile", type=str, help="Path to the standard error log file")
    parser.add_argument("datasetname", type=str, help="Name of the Dataset class in kd_data_sets")
    parser.add_argument("datafolder", type=str, help="Name of the folder in generated_datasets")
    parser.add_argument("savename", type=str, help="Name of model save")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (optional)")
    parser.add_argument("--resume", action="store_true", help="Resume training from a saved model")
    parser.add_argument("--clip", action="store_true", help="apply gradient clipping at 1.0")
    parser.add_argument("--he", action="store_true", help="performs he init")
    parser.add_argument("--CE", action="store_true", help="performs Cross Entropy and MSE as loss")
    parser.add_argument("--ceweight", type=float, default=1.0, help="multiplies the Cross Entropy loss (changes relative weight")
    args = parser.parse_args()



    train_model(args.modelname, args.kwargsfile, args.epochs, args.datasetname,args.datafolder, args.savename, args.lr, args.resume, args.clip, args.he, args.CE, args.ceweight)

    

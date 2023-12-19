import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Dataset
import os
import random


class LlamaDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.input_datasets = []
        self.target_datasets = []
        file_ids = []
    
   
        all_files = os.listdir(data_dir)

        for file_name in all_files:
            if file_name.startswith("input") and file_name.endswith(".pt"):
                file_id = file_name[len("input"):-len(".pt")]
                target_file_name = f"target{file_id}.pt"

                # Check if there is a corresponding target file
                if target_file_name in all_files:
                    #if int(file_id) < 10: ##this is to just load 50 files.  i need to make it so it loads subsets i think
                    file_ids.append(int(file_id))
                    
        self.file_ids = file_ids
        
        for id in file_ids:
            input_file_path = os.path.join(data_dir, f'input{id}.pt')
            target_file_path = os.path.join(data_dir, f'target{id}.pt')
            
            # Load input and target tensors
            input_tensor = torch.load(input_file_path).to(torch.float32)
            target_tensor = torch.load(target_file_path)
            
            # Create TensorDataset for input and target tensors
            input_dataset = TensorDataset(input_tensor)
            target_dataset = TensorDataset(target_tensor)
            
            # Append the datasets to the lists
            self.input_datasets.append(input_dataset)
            self.target_datasets.append(target_dataset)

        # Combine the individual datasets into a single dataset using ConcatDataset
        self.combined_input_dataset = ConcatDataset(self.input_datasets)
        self.combined_target_dataset = ConcatDataset(self.target_datasets)

    def __len__(self):
        return len(self.combined_input_dataset)

    def __getitem__(self, idx):
        return (
            self.combined_input_dataset[idx][0],  # Get the input tensor
            self.combined_target_dataset[idx][0]  # Get the target tensor
        )

class LowMemLlamaDataset(Dataset):
    def __init__(self, data_dir, batch_size = 10):
        self.data_dir = data_dir
        self.file_ids = []
        self.total_length = 0  # Initialize the total length

        all_files = os.listdir(data_dir)

        for file_name in all_files:
            if file_name.startswith("input") and file_name.endswith(".pt"):
                file_id = file_name[len("input"):-len(".pt")]
                target_file_name = f"target{file_id}.pt"

                # Check if there is a corresponding target file
                if target_file_name in all_files:
                    if int(file_id) < 10:  # Adjust this condition as needed for your subset
                        self.file_ids.append(int(file_id))

                        # Calculate the size of the first dimension of the tensors in this file
                        input_file_path = os.path.join(data_dir, f'input{file_id}.pt')
                        input_tensor = torch.load(input_file_path)
                        self.total_length += input_tensor.size(0)

    def __len__(self):
        return self.total_length  # Return the total length based on the first dimension

    def __getitem__(self, idx):
        # Calculate which file and position within the file corresponds to the given index
        for file_id in self.file_ids:
            input_file_path = os.path.join(self.data_dir, f'input{file_id}.pt')
            input_tensor = torch.load(input_file_path)
            file_length = input_tensor.size(0)

            if idx < file_length:
                target_file_path = os.path.join(self.data_dir, f'target{file_id}.pt')
                target_tensor = torch.load(target_file_path)
                return input_tensor[idx], target_tensor[idx]

            idx -= file_length

class SpeedyLlamaDataset(Dataset):
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.file_ids = []

        all_files = os.listdir(data_dir)

        for file_name in all_files:
            if file_name.startswith("input") and file_name.endswith(".pt"):
                file_id = file_name[len("input"):-len(".pt")]
                target_file_name = f"target{file_id}.pt"

                # Check if there is a corresponding target file
                if target_file_name in all_files:
                    if int(file_id) < 10:  # Adjust this condition as needed for your subset
                        self.file_ids.append(int(file_id))

        self.batch_size = batch_size

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        input_file_path = os.path.join(self.data_dir, f'input{file_id}.pt')
        target_file_path = os.path.join(self.data_dir, f'target{file_id}.pt')

        # Load input and target tensors when requested
        input_tensor = torch.load(input_file_path).to(torch.float32)
        target_tensor = torch.load(target_file_path)

        return input_tensor, target_tensor  
    
def create_mapping(vocab_mask):
    mapping = {new_index: old_index for old_index, new_index in enumerate(vocab_mask)}
    return mapping

#https://chat.openai.com/c/04aeae04-ad2f-49c2-a32c-3ce5159fadd9 do the alternating thing.  
class DoubleFileDataset(Dataset): #could make variable file?
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_ids = []
        self.total_length = 0  # Initialize the total length
        self.current_file_1 = None
        self.current_target_1 = None
        self.current_file_2 = None
        self.current_target_2 = None
        self.current_file_1_ids = []
        self.current_file_2_ids = []
        self.used_file_ids = set()
        
        all_files = os.listdir(data_dir)

        for file_name in all_files:
            if file_name.startswith("input") and file_name.endswith(".pt"): #looks for formatted pt files. 
                file_id = file_name[len("input"):-len(".pt")]
                target_file_name = f"target{file_id}.pt"
                
                file_size = 0
                # Check if there is a corresponding target file
                if target_file_name in all_files:
                    if int(file_id) < 20:  # Adjust this condition as needed for your subset #remove for real thing
                        print(file_id)
                        self.file_ids.append(int(file_id)) #appends only if there is a target and input file
                        if file_size == 0:
                            input_file_path = os.path.join(data_dir, f'input{file_id}.pt')
                            input_tensor = torch.load(input_file_path)
                            file_size = input_tensor.size(0) #this is loading em, but replaces.  not sure it ruins
                        self.total_length += file_size 
                        
        print(f"Training on {len(self.file_ids)} file(s) and {self.total_length} sample(s).")
        self.load_next_file()

    def load_next_file(self):
        if self.current_file_1 is None:
            self.current_file_1, self.current_target_1 = self.load_random_file()
            self.current_file_1_ids = list(range(len(self.current_file_1)))
        elif self.current_file_2 is None:
            self.current_file_2, self.current_target_2 = self.load_random_file()
            self.current_file_2_ids = list(range(len(self.current_file_2)))

    def load_random_file(self):
        # Choose a random file
        unused_file_ids = list(set(self.file_ids) - self.used_file_ids)
        if not unused_file_ids:
            # Reset used_file_ids for the next epoch
            self.used_file_ids = set()
            unused_file_ids = self.file_ids
        
        file_id = random.choice(unused_file_ids)
        #print(f'loading file {file_id}')
        self.used_file_ids.add(file_id)
        input_file_path = os.path.join(self.data_dir, f'input{file_id}.pt')
        output_file_path = os.path.join(self.data_dir, f'target{file_id}.pt')
        input_tensor = torch.load(input_file_path)
        output_tensor = torch.load(output_file_path)
        return input_tensor, output_tensor

    def __len__(self):
        return self.total_length  # Return the total length based on the first dimension

    def __getitem__(self, idx):
        if not self.current_file_1_ids:
            self.current_file_1 = None
            #print('got here1')
            self.load_next_file()
        if not self.current_file_2_ids:
            #print('got here2')
            self.current_file_2 = None
            self.load_next_file()
        
        # Choose between current_file_1 and current_file_2 randomly
        current_file = self.current_file_1 if random.random() < 0.5 else self.current_file_2
        current_ids = self.current_file_1_ids if current_file is self.current_file_1 else self.current_file_2_ids
        current_targets = self.current_target_1 if current_file is self.current_file_1 else self.current_target_2
        #print(current_ids)
        # Randomly select an item from the chosen file
        selected_idx = random.choice(current_ids)
        #print(f'adding id {selected_idx}')
        current_ids.remove(selected_idx)
        
        return current_file[selected_idx], current_targets[selected_idx]  # Return input and target

    
class CPUDoubleFileDataset(Dataset): #could make variable file?
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_ids = []
        self.total_length = 0  # Initialize the total length
        self.current_file_1 = None
        self.current_target_1 = None
        self.current_file_2 = None
        self.current_target_2 = None
        self.current_file_1_ids = []
        self.current_file_2_ids = []
        self.used_file_ids = set()
        
        all_files = os.listdir(data_dir)

        for file_name in all_files:
            if file_name.startswith("input") and file_name.endswith(".pt"): #looks for formatted pt files. 
                file_id = file_name[len("input"):-len(".pt")]
                target_file_name = f"target{file_id}.pt"
                
                file_size = 0
                # Check if there is a corresponding target file
                if target_file_name in all_files:
                    if int(file_id) < 2000:  # Adjust this condition as needed for your subset #remove for real thing
                        print(file_id)
                        self.file_ids.append(int(file_id)) #appends only if there is a target and input file
                        if file_size == 0:
                            input_file_path = os.path.join(data_dir, f'input{file_id}.pt')
                            input_tensor = torch.load(input_file_path, map_location=torch.device('cpu'))
                            file_size = input_tensor.size(0) #this is loading em, but replaces.  not sure it ruins
                        self.total_length += file_size 
                        
        print(f"Training on {len(self.file_ids)} file(s) and {self.total_length} sample(s).")
        self.load_next_file()

    def load_next_file(self):
        if self.current_file_1 is None:
            self.current_file_1, self.current_target_1 = self.load_random_file()
            self.current_file_1_ids = list(range(len(self.current_file_1)))
        elif self.current_file_2 is None:
            self.current_file_2, self.current_target_2 = self.load_random_file()
            self.current_file_2_ids = list(range(len(self.current_file_2)))

    def load_random_file(self):
        # Choose a random file
        unused_file_ids = list(set(self.file_ids) - self.used_file_ids)
        if not unused_file_ids:
            # Reset used_file_ids for the next epoch
            self.used_file_ids = set()
            unused_file_ids = self.file_ids
        
        file_id = random.choice(unused_file_ids)
        #print(f'loading file {file_id}')
        self.used_file_ids.add(file_id)
        input_file_path = os.path.join(self.data_dir, f'input{file_id}.pt')
        output_file_path = os.path.join(self.data_dir, f'target{file_id}.pt')
        input_tensor = torch.load(input_file_path, map_location=torch.device('cpu')).to(torch.float32)
        output_tensor = torch.load(output_file_path, map_location=torch.device('cpu')).to(torch.float32)
        return input_tensor, output_tensor

    def __len__(self):
        return self.total_length  # Return the total length based on the first dimension

    def __getitem__(self, idx):
        if not self.current_file_1_ids:
            self.current_file_1 = None
            #print('got here1')
            self.load_next_file()
        if not self.current_file_2_ids:
            #print('got here2')
            self.current_file_2 = None
            self.load_next_file()
        
        # Choose between current_file_1 and current_file_2 randomly
        current_file = self.current_file_1 if random.random() < 0.5 else self.current_file_2
        current_ids = self.current_file_1_ids if current_file is self.current_file_1 else self.current_file_2_ids
        current_targets = self.current_target_1 if current_file is self.current_file_1 else self.current_target_2
        #print(current_ids)
        # Randomly select an item from the chosen file
        selected_idx = random.choice(current_ids)
        #print(f'adding id {selected_idx}')
        current_ids.remove(selected_idx)
        
        return current_file[selected_idx], current_targets[selected_idx]  # Return input and target

class CacheDoubleFileDataset(Dataset):
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.file_ids = []
        self.batch_size = batch_size
        self.data_cache = []

        all_files = os.listdir(data_dir)

        for file_name in all_files:
            if file_name.startswith("input") and file_name.endswith(".pt"):
                file_id = file_name[len("input"):-len(".pt")]
                target_file_name = f"target{file_id}.pt"

                if target_file_name in all_files:
                    self.file_ids.append(int(file_id))

        print(f"Training on {len(self.file_ids)} file(s).")

    def load_data(self):
        random.shuffle(self.file_ids)
        self.data_cache = []

        for file_id in self.file_ids:
            input_file_path = os.path.join(self.data_dir, f'input{file_id}.pt')
            output_file_path = os.path.join(self.data_dir, f'target{file_id}.pt')
            input_tensor = torch.load(input_file_path, map_location=torch.device('cpu')).to(torch.float32)
            output_tensor = torch.load(output_file_path, map_location=torch.device('cpu')).to(torch.float32)
            self.data_cache.append((input_tensor, output_tensor))

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        if len(self.data_cache) == 0:
            self.load_data()

        input_data, output_data = self.data_cache.pop()

        return input_data, output_data

class MaskedLlamaDataset(Dataset): #this needs to do the masks on DoubleFile
    def __init__(self, data_dir, mask):
        self.data_dir = data_dir
        self.input_datasets = []
        self.target_datasets = []
        if not isinstance(mask, set):
            raise ValueError("vocab_mask_set should be a set")
        self.vocab_mask = sorted(list(mask))
        self.mask_mapping = create_mapping(self.vocab_mask)
        file_ids = []
    
   
        all_files = os.listdir(data_dir)

        for file_name in all_files:
            if file_name.startswith("input") and file_name.endswith(".pt"):
                file_id = file_name[len("input"):-len(".pt")]
                target_file_name = f"target{file_id}.pt"

                # Check if there is a corresponding target file
                if target_file_name in all_files:
                    file_ids.append(int(file_id))
                    
        self.file_ids = file_ids
        
        for id in file_ids:
            input_file_path = os.path.join(data_dir, f'input{id}.pt')
            target_file_path = os.path.join(data_dir, f'target{id}.pt')
            
            # Load input and target tensors
            input_tensor = torch.load(input_file_path)
            target_tensor = torch.load(target_file_path)
            
            #mask em
            input_tensor_masked = input_tensor[:, :, self.vocab_mask]
            target_tensor_masked = target_tensor[:, self.vocab_mask]

            
            # Create TensorDataset for input and target tensors
            input_dataset = TensorDataset(input_tensor_masked)
            target_dataset = TensorDataset(target_tensor_masked)
            
            # Append the datasets to the lists
            self.input_datasets.append(input_dataset)
            self.target_datasets.append(target_dataset)

        # Combine the individual datasets into a single dataset using ConcatDataset
        self.combined_input_dataset = ConcatDataset(self.input_datasets)
        self.combined_target_dataset = ConcatDataset(self.target_datasets)

    def __len__(self):
        return len(self.combined_input_dataset)

    def __getitem__(self, idx):
        return (
            self.combined_input_dataset[idx][0],  # Get the input tensor
            self.combined_target_dataset[idx][0]  # Get the target tensor
        )



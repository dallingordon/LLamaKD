import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import math #only for Transformerbaby, and with vocab size and seq_len, untenable.  
##this is all wrong.  cross product between the words makes no sense lolol
#a sentence embedding tho.

class LlamaFetus(nn.Module):
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , hidden_dim #vestigial.  fix this
                 , word_embed
                 , sentence_embed
                 , balanced_dim
                ):
        super(LlamaFetus, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        self.sentence_embed = int(sentence_embed)
        self.balanced_dim = int(balanced_dim)
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) #50 is word embeddings essentially
        self.sentence_embedding = nn.Linear(self.sequence_length,self.sentence_embed) #also could apply twice.
        self.we_down = nn.Linear(self.word_embed,self.balanced_dim)
        self.seq_down = nn.Linear(self.sentence_embed,self.balanced_dim) #could apply twice
    
        self.out = nn.Linear(self.balanced_dim**3,self.vocab_size)
        # Activation function

    def forward(self, x):
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        x = x.permute(0, 2, 1)
        x = F.relu(self.sentence_embedding(x))
        x = x.permute(0, 2, 1)
        # Flatten the sequence_length and vocab_size dimensions
        x = F.relu(torch.einsum('bij,bkm->bikj', x, x)) #interactions
        x = F.relu(self.we_down(x))
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.seq_down(x))
        x = x.permute(0, 1, 3, 2)
        x = F.relu(self.seq_down(x)) #duplicated, could be a second layer.   
        x = x.view(x.shape[0],-1)
        # Pass through the second fully connected layer
        x = self.out(x)
        #x = F.softmax(x,dim=1)
        return x
    

class DimMemory(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, memory_dim):
        super(DimMemory, self).__init__()
        if memory_dim < 1:
            raise ValueError("memory_dim must be greater than or equal to 1.")
        
        self.memory_dim = memory_dim 
        # Create memory tensor with dynamic shape
        mem_args = [1]
        mem_args.extend([hidden_dim for _ in range(memory_dim)])
        self.mem = nn.Parameter(torch.randn(*mem_args))
        self.mem.requires_grad = True
        
        # Create a list of linear layers with memory_dim - 1 repetitions
        self.linears = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(memory_dim - 1)
        ])

        # Create the final linear layer for output
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input1):
        # Expand memory tensor along the third dimension
        mem_forward_args = [input1.shape[0]]
        mem_forward_args.extend([-1 for _ in range(self.memory_dim)])
        
        x = self.mem.expand(*mem_forward_args)
        
        for i, linear_layer in enumerate(self.linears):
            #print(linear_layer.weight.shape)
            y = torch.relu(linear_layer(input1)) #.squeeze() I don't think i need this
            #print(y.shape, x.shape)
            x = torch.einsum('az,a...yz->a...y',y,x)
            #print(x.shape)
        # Apply the final linear layer for output
        #print(x.shape)
        x = torch.relu(x) 
        x = torch.relu(self.linear_out(x))
        #print(x.shape)
        return x
    
class MemoryLlama(nn.Module):
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , hidden_dim
                 , word_embed
                 , sentence_embed
                 , balanced_dim
                 , mem_input_dim
                 , mem_hidden_dim
                 , mem_output_dim
                 , memory_dim
                ):
        super(MemoryLlama, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        self.sentence_embed = int(sentence_embed)
        self.balanced_dim = int(balanced_dim)
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) #50 is word embeddings essentially
        self.sentence_embedding = nn.Linear(self.sequence_length,self.sentence_embed) #also could apply twice.
        self.we_down = nn.Linear(self.word_embed,self.balanced_dim)
        self.seq_down = nn.Linear(self.sentence_embed,self.balanced_dim) #could apply twice
        self.to_mem = nn.Linear(self.balanced_dim**3,mem_input_dim)
        self.dim_memory = DimMemory(mem_input_dim,mem_hidden_dim,mem_output_dim,memory_dim)
        
        self.out = nn.Linear(mem_input_dim + mem_output_dim,self.vocab_size)
        # Activation function

    def forward(self, x):
        
        x = torch.relu(self.word_embedding(x)) #sentence of word embeddings.  
        x = x.permute(0, 2, 1)
        x = torch.relu(self.sentence_embedding(x))
        x = x.permute(0, 2, 1)
        # Flatten the sequence_length and vocab_size dimensions
        x = torch.relu(torch.einsum('bij,bkm->bikj', x, x)) #interactions
        x = torch.relu(self.we_down(x))
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.seq_down(x))
        x = x.permute(0, 1, 3, 2)
        x = torch.relu(self.seq_down(x)) #duplicated, could be a second layer.   
        
        x = x.view(x.shape[0],-1)
        x = torch.relu(self.to_mem(x))
        y = self.dim_memory(x)
        x = torch.cat((x, y), dim=1)
        #print(x.shape)
        x = self.out(x)
        #x = F.softmax(x,dim=1)
        return x

class LlamaBaby(nn.Module):
    """This flattens everything at the end so you have balanced_dim ** 3 in the second to last layer"""
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , hidden_dim #vestigial lololol
                 , word_embed
                 , sentence_embed
                 , balanced_dim
                ):
        super(LlamaBaby, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        self.sentence_embed = int(sentence_embed)
        self.balanced_dim = int(balanced_dim)
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) #50 is word embeddings essentially
        self.sentence_embedding = nn.Linear(self.sequence_length,self.sentence_embed) #also could apply twice.
        self.we_down = nn.Linear(self.word_embed,self.balanced_dim)
        self.seq_down = nn.Linear(self.sentence_embed,self.balanced_dim) #could apply twice
        self.out_down = nn.Linear(self.sentence_embed,self.balanced_dim)
        self.out = nn.Linear(self.balanced_dim**3,self.vocab_size)
        # Activation function

    def forward(self, x):
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        x = x.permute(0, 2, 1)
        x = F.relu(self.sentence_embedding(x))
        x = x.permute(0, 2, 1)
        x = F.relu(torch.einsum('bij,bkm->bikj', x, x)) #interactions
        x = F.relu(self.we_down(x))
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.seq_down(x))
        x = x.permute(0, 1, 3, 2)
        x = F.relu(self.out_down(x)) 
        x = x.view(x.shape[0],-1)
        x = self.out(x)
        return x
    
class MemoryBaby(nn.Module):
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , hidden_dim
                 , word_embed
                 , sentence_embed
                 , balanced_dim
                 , mem_input_dim
                 , mem_hidden_dim
                 , mem_output_dim
                 , memory_dim
                ):
        super(MemoryBaby, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        self.sentence_embed = int(sentence_embed)
        self.balanced_dim = int(balanced_dim)
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) #50 is word embeddings essentially
        self.sentence_embedding = nn.Linear(self.sequence_length,self.sentence_embed) #also could apply twice.
        self.we_down = nn.Linear(self.word_embed,self.balanced_dim)
        self.seq_down = nn.Linear(self.sentence_embed,self.balanced_dim)
        self.out_down = nn.Linear(self.sentence_embed,self.balanced_dim)
        self.to_mem = nn.Linear(self.balanced_dim**3,mem_input_dim)
        self.dim_memory = DimMemory(mem_input_dim,mem_hidden_dim,mem_output_dim,memory_dim)
        
        self.out = nn.Linear(mem_input_dim + mem_output_dim,self.vocab_size)
        # Activation function

    def forward(self, x):
        
        x = torch.relu(self.word_embedding(x)) #sentence of word embeddings.  
        x = x.permute(0, 2, 1)
        x = torch.relu(self.sentence_embedding(x))
        x = x.permute(0, 2, 1)
        # Flatten the sequence_length and vocab_size dimensions
        x = torch.relu(torch.einsum('bij,bkm->bikj', x, x)) #interactions
        x = torch.relu(self.we_down(x))
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.seq_down(x))
        x = x.permute(0, 1, 3, 2)
        x = torch.relu(self.out_down(x)) #duplicated, could be a second layer.   
        
        x = x.view(x.shape[0],-1)
        x = torch.relu(self.to_mem(x))
        y = self.dim_memory(x)
        x = torch.cat((x, y), dim=1)
        x = self.out(x)
        #x = F.softmax(x,dim=1)
        return x
    

#
class TransformerBaby(nn.Module):
    """
    A transformer model that accepts one-hot encoded input and outputs a tensor of shape batch x vocab_size.
    """
    def __init__(self, vocab_size
                 , sequence_length
                 , d_model
                 , nhead
                 , num_layers
                 , dim_feedforward
                ):
        super(TransformerBaby, self).__init__()

        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.d_model = d_model

        # Linear layer to match d_model size
        self.word_embedding = nn.Linear(vocab_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, sequence_length)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Final Linear Layer
        self.fc = nn.Linear(d_model * sequence_length, vocab_size)

    def forward(self, src):
        src = self.word_embedding(src)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)
        output = output.view(output.size(0), -1)  # Flattening
        output = self.fc(output)

        return output

class PositionalEncoding(nn.Module):
    """
    Positional Encoding that adds position information to input embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        x = x + self.encoding[:, :x.size(1)]
        return x
    
class CrossBaby_1(nn.Module):
    
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                ):
        super(CrossBaby_1, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        self.reduce = nn.Linear(self.sequence_length*self.word_embed,self.word_embed)
        self.reduce_2 = nn.Linear(self.sequence_length*self.word_embed,self.word_embed)
        self.out = nn.Linear(self.word_embed,self.vocab_size)
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape)
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "after reshape")
        x = F.relu(self.reduce(x))
        #print(x.shape, "after reduce")
        x = x.reshape(x.shape[0],-1)
        #print(x.shape, "after reshape 2")
        x = F.relu(self.reduce_2(x))
        #print(x.shape,"after reduce_2")
        x = self.out(x)
        return x
    
class CrossBaby_2(nn.Module):
    
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                 , up_dim
                ):
        super(CrossBaby_2, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        
        self.x_a_linear = nn.Linear(self.word_embed*self.sequence_length,self.word_embed)
        self.a_down = nn.Linear(self.word_embed*self.sequence_length,self.word_embed )
        
        self.x_b_linear = nn.Linear(self.word_embed*self.sequence_length,self.word_embed)
        self.b_down = nn.Linear(self.word_embed*self.sequence_length,self.word_embed )
        
        self.x_c_linear = nn.Linear(self.sequence_length*self.sequence_length,self.word_embed)
        self.c_down = nn.Linear(self.word_embed*self.word_embed,self.word_embed )
        
        self.out_1 = nn.Linear(3*self.word_embed,up_dim)
        self.out_2 = nn.Linear(up_dim,vocab_size )
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape)
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x_a = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x_a.shape, "x_a")
        a = F.relu(self.x_a_linear(x_a))
        a = a.reshape(a.shape[0],-1)
        a = F.relu(self.a_down(a))
        #print(a.shape,"a")
        x_b = x.permute(0,2,1,3).reshape(x.shape[0],x.shape[1],-1)
        #print(x_b.shape, "x_b")
        b = F.relu(self.x_b_linear(x_b))
        b = b.reshape(b.shape[0],-1)
        b = F.relu(self.b_down(b))
        #print(b.shape, "b")
        x_c = x.permute(0,3,1,2).reshape(x.shape[0],x.shape[-1],-1)
        #print(x_c.shape, "x_c")
        c = F.relu(self.x_c_linear(x_c))
        c = c.reshape(c.shape[0],-1)
        c = F.relu(self.c_down(c))
        #print(c.shape)
        d = torch.concat((a,b,c), dim=-1)
        #print(d.shape)
        d = F.relu(self.out_1(d))
        d = self.out_2(d)
        return d
    
class SquareMemory(nn.Module):
    def __init__(self,input_dim, memory_dim):
        super(SquareMemory, self).__init__()
        
        self.mem = nn.Parameter(torch.randn(memory_dim,memory_dim))
        self.mem.requires_grad = True
        
        

        # Create the final linear layer for output
        self.first_axis = nn.Linear(input_dim, memory_dim)
        self.second_axis = nn.Linear(input_dim, memory_dim)
        self.intercept = nn.Linear(input_dim,memory_dim)

    def forward(self, input1):
        #batch_dim = input1.shape[0]
        #print(batch_dim)
        first = F.softmax(self.first_axis(input1), dim=-1)
        first = torch.matmul(first, self.mem) #first is used as the lookup, picks a row with the softmax
        
        second = F.relu(self.second_axis(input1))
        
        x = second * first
        
        intercept = F.relu(self.intercept(input1))
        x = x + intercept
        return x
    
    
class SMCrossBaby_1(nn.Module):
    """This flattens everything at the end so you have balanced_dim ** 3 in the second to last layer"""
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                 , memory_dim
                ):
        super(SMCrossBaby_1, self).__init__()
        
        self.mem = SquareMemory(word_embed,memory_dim)
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        self.reduce = nn.Linear(self.sequence_length*self.word_embed,self.word_embed)
        self.reduce_2 = nn.Linear(self.sequence_length*self.word_embed,self.word_embed)
        
        
        
        self.out = nn.Linear(memory_dim,self.vocab_size)
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape)
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "after reshape")
        x = F.relu(self.reduce(x))
        #print(x.shape, "after reduce")
        x = x.reshape(x.shape[0],-1)
        #print(x.shape, "after reshape 2")
        x = F.relu(self.reduce_2(x))
        #print(x.shape,"after reduce_2")
        x = self.mem(x)
        #print(x.shape, "memory")
        x = self.out(x)
        return x
    
class SMCrossBaby_Concat(nn.Module):

    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                 , memory_dim
                ):
        super(SMCrossBaby_Concat, self).__init__()
        
        self.mem = SquareMemory(word_embed,memory_dim)
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        self.reduce = nn.Linear(self.sequence_length*self.word_embed,self.word_embed)
        self.reduce_2 = nn.Linear(self.sequence_length*self.word_embed,self.word_embed)
        self.conc = nn.Linear(self.sequence_length*self.word_embed,memory_dim)
        
        
        self.out = nn.Linear(memory_dim*2,self.vocab_size)
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape)
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "after reshape")
        x = F.relu(self.reduce(x))
        #print(x.shape, "after reduce")
        x = x.reshape(x.shape[0],-1)
        #print(x.shape, "after reshape 2")
        c = F.relu(self.conc(x))
        x = F.relu(self.reduce_2(x))
        #print(x.shape,"after reduce_2")
        x = self.mem(x)
        x = torch.concat((x, c), dim = -1)
        #print(x.shape, 'after concat')
        x = self.out(x)
        return x
    
class DoubleCrossBaby(nn.Module):
    """2 interactions in the embedding space.  then sums.  that sum could be learned?"""
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                ):
        super(DoubleCrossBaby, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        self.reduce = nn.Linear(self.sequence_length*self.word_embed,self.word_embed*2)
        self.reduce_2 = nn.Linear(self.sequence_length*self.word_embed*2,self.word_embed*5)
        
        self.up_1 = nn.Linear(self.word_embed*5, self.word_embed*10)
        self.up_2 = nn.Linear(self.word_embed*10,self.vocab_size)
        
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape)
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "after reshape")
        x = F.relu(self.reduce(x))
        #print(x.shape, "after reduce")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape,"second ein")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "another reshape")
        x = F.relu(self.reduce_2(x))
        #print(x.shape, "second reduce")
        x = F.relu(self.up_1(x))
        #print(x.shape, "up_1")
        x = F.relu(self.up_2(x))
        #print(x.shape,"up_2")
        x = torch.sum(x, dim=1)
        #print(x.shape)
        return x
    
class LearnedDoubleCrossBaby(nn.Module):
    """2 interactions in the embedding space.  leanred vocab agg"""
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                ):
        super(LearnedDoubleCrossBaby, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        self.reduce = nn.Linear(self.sequence_length*self.word_embed,self.word_embed*2)
        self.reduce_2 = nn.Linear(self.sequence_length*self.word_embed*2,self.word_embed*5)
        
        self.up_1 = nn.Linear(self.word_embed*5, self.word_embed*10)
        self.up_2 = nn.Linear(self.word_embed*10,self.vocab_size)
        self.final = nn.Linear(self.sequence_length, 1)
        
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape)
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "after reshape")
        x = F.relu(self.reduce(x))
        #print(x.shape, "after reduce")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape,"second ein")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "another reshape")
        x = F.relu(self.reduce_2(x))
        #print(x.shape, "second reduce")
        x = F.relu(self.up_1(x))
        #print(x.shape, "up_1")
        x = F.relu(self.up_2(x))
        #print(x.shape,"up_2")
        x = x.transpose(1, 2)
        x = F.relu(self.final(x))
        x = x.squeeze(-1)
        return x
    
def int_to_binary_tensor(number, max_length):

    binary_string = bin(number)[2:]  # Convert to binary and remove the '0b' prefix.
    same_len = binary_string.zfill(max_length)
    seperated =torch.tensor([float(i) for i in same_len])
    return seperated
def create_binary_tensor(input_length):
    max_length = math.ceil(math.log2(input_length + 1))
    binary_tensors = []

    # Iterate through numbers from 1 to input_length
    for number in range(1, input_length + 1):
        binary_tensor = int_to_binary_tensor(number, max_length)
        #print(binary_tensor.shape)
        binary_tensors.append(binary_tensor)
        
    # Stack the binary tensors to form a 2D tensor
    stacked_tensor = torch.stack(binary_tensors)

    return stacked_tensor

def binary_self_interactions(input_length):
    stacked_tensor = create_binary_tensor(input_length)
    res = torch.zeros((stacked_tensor.shape[0],stacked_tensor.shape[1]**2))
    
    for i in range(stacked_tensor.shape[0]):
        res[i, :] = torch.outer(stacked_tensor[i],stacked_tensor[i]).flatten()
    return res

class BinaryPositionalEmbedding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(BinaryPositionalEmbedding, self).__init__()
        
        self.positional_input = binary_self_interactions(max_len)
        
        # Determine positional_emb_dim from positional_input
        positional_emb_dim = self.positional_input.shape[1]
        
        # Linear layer with input size of positional_emb_dim and output size of embedding_dim
        self.linear = nn.Linear(positional_emb_dim, embedding_dim)

    def forward(self, x):
        
        batch_size = x.shape[0]
        # Apply the linear layer to each position
        x_add = F.relu(self.linear(self.positional_input))
        
        x_add = x_add.unsqueeze(0).repeat(batch_size, 1, 1)
        
        
        return x + x_add


class LearnedDoubleCrossBabyWithBinaryEmbedding(nn.Module):
    """2 interactions in the embedding space.  then sums.  that sum could be learned?"""
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                ):
        super(LearnedDoubleCrossBabyWithBinaryEmbedding, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        self.word_pos_emb = BinaryPositionalEmbedding(self.sequence_length,self.word_embed)
        
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        self.reduce = nn.Linear(self.sequence_length*self.word_embed,self.word_embed*2)
        self.word_pos_emb_reduce = BinaryPositionalEmbedding(sequence_length,self.word_embed*2)
        
        self.reduce_2 = nn.Linear(self.sequence_length*self.word_embed*2,self.word_embed*5)
        self.word_pos_emb_reduce_2 = BinaryPositionalEmbedding(sequence_length,self.word_embed*5)
        
        self.up_1 = nn.Linear(self.word_embed*5, self.word_embed*10)
        self.up_2 = nn.Linear(self.word_embed*10,self.vocab_size)
        self.final = nn.Linear(self.sequence_length, 1)
        
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape, "first embedding")
        #add pos emb here
        #print(x.device)
        x = self.word_pos_emb(x)
        #print(x.shape, "after embedding")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "after reshape")
        x = F.relu(self.reduce(x))
        x = self.word_pos_emb_reduce(x)
        #add pos emb here
        #print(x.shape, "after reduce first cross")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape,"second ein")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "another reshape")
        x = F.relu(self.reduce_2(x))
        x = self.word_pos_emb_reduce_2(x)
        #as pos emb here
        #print(x.shape, "second reduce")
        x = F.relu(self.up_1(x))
        #print(x.shape, "up_1")
        x = F.relu(self.up_2(x))
        #print(x.shape,"up_2")
        x = x.transpose(1, 2)
        x = F.relu(self.final(x))
        x = x.squeeze(-1)
        return x
    
    def to(self, device):
        self = super().to(device)
        self.word_pos_emb.positional_input = self.word_pos_emb.positional_input.to(device)
        self.word_pos_emb_reduce.positional_input = self.word_pos_emb_reduce.positional_input.to(device)
        self.word_pos_emb_reduce_2.positional_input = self.word_pos_emb_reduce_2.positional_input.to(device)
        return self
    
def getPositionEncoding(seq_len, d, n=10000):
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in range(int(d / 2)):
            denominator = torch.pow(torch.tensor(n, dtype=torch.float32), 2 * i / d)
            P[k, 2 * i] = torch.sin(k / denominator)
            P[k, 2 * i + 1] = torch.cos(k / denominator)
    return P

class LearnedSinPositionalEmbedding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(LearnedSinPositionalEmbedding, self).__init__()
         
        self.positional_input = getPositionEncoding(seq_len=max_len, d=embedding_dim*2, n=10_000) #10k is from attention is all...
        
        # Determine positional_emb_dim from positional_input
        positional_emb_dim = self.positional_input.shape[1]
        
        # Linear layer with input size of positional_emb_dim and output size of embedding_dim
        self.linear = nn.Linear(positional_emb_dim, embedding_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        # Apply the linear layer to each position
        x_add = F.relu(self.linear(self.positional_input))
        x_add = x_add.unsqueeze(0).repeat(batch_size, 1, 1)
        
        
        return x + x_add
    
class SinPositionalEmbedding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(SinPositionalEmbedding, self).__init__()
        
        self.positional_input = getPositionEncoding(seq_len=max_len, d=embedding_dim, n=10_000)

    def forward(self, x):
        batch_size = x.shape[0]

        x_add = self.positional_input.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return x + x_add
    
class LearnedDoubleCrossBabyWithSinEmbedding(nn.Module):
    """2 interactions in the embedding space.  then sums.  that sum could be learned?"""
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                ):
        super(LearnedDoubleCrossBabyWithSinEmbedding, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        self.word_pos_emb = SinPositionalEmbedding(self.sequence_length,self.word_embed)
        
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        self.reduce = nn.Linear(self.sequence_length*self.word_embed,self.word_embed*2)
        self.word_pos_emb_reduce = SinPositionalEmbedding(sequence_length,self.word_embed*2)
        
        self.reduce_2 = nn.Linear(self.sequence_length*self.word_embed*2,self.word_embed*5)
        self.word_pos_emb_reduce_2 = SinPositionalEmbedding(sequence_length,self.word_embed*5)
        
        self.up_1 = nn.Linear(self.word_embed*5, self.word_embed*10)
        self.up_2 = nn.Linear(self.word_embed*10,self.vocab_size)
        self.final = nn.Linear(self.sequence_length, 1)
        
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape, "first embedding")
        #add pos emb here
        x = self.word_pos_emb(x)
        #print(x.shape, "after embedding")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "after reshape")
        x = F.relu(self.reduce(x))
        x = self.word_pos_emb_reduce(x)
        #add pos emb here
        #print(x.shape, "after reduce first cross")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape,"second ein")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "another reshape")
        x = F.relu(self.reduce_2(x))
        x = self.word_pos_emb_reduce_2(x)
        #as pos emb here
        #print(x.shape, "second reduce")
        x = F.relu(self.up_1(x))
        #print(x.shape, "up_1")
        x = F.relu(self.up_2(x))
        #print(x.shape,"up_2")
        x = x.transpose(1, 2)
        x = F.relu(self.final(x))
        x = x.squeeze(-1)
        return x
    
    def to(self, device):
        self = super().to(device)
        self.word_pos_emb.positional_input = self.word_pos_emb.positional_input.to(device)
        self.word_pos_emb_reduce.positional_input = self.word_pos_emb_reduce.positional_input.to(device)
        self.word_pos_emb_reduce_2.positional_input = self.word_pos_emb_reduce_2.positional_input.to(device)
        return self
    
class LearnedDoubleCrossBabyWithLearnedSinEmbedding(nn.Module):
    """2 interactions in the embedding space.  then sums.  that sum could be learned?"""
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                 , first_reduce = None
                 , second_reduce = None
                 , up_scale = None
                ):
        super(LearnedDoubleCrossBabyWithLearnedSinEmbedding, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        
        #deal with defaults for backwards compatibility 
        #i initially made this so it could scale just by changing the word_embed, thus the multiplication
        #now we growing these I want more control so:
        
        if first_reduce is None:
            self.first_reduce = self.word_embed*2
        else:
            self.first_reduce = int(first_reduce)
        
        if second_reduce is None:
            self.second_reduce = self.word_embed*5
        else:
            self.second_reduce = int(second_reduce)
        
        if up_scale is None:
            self.up_scale = self.word_embed*10
        else:
            self.up_scale = int(up_scale)
        
        self.word_pos_emb = LearnedSinPositionalEmbedding(self.sequence_length,self.word_embed)
        
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        self.reduce = nn.Linear(self.sequence_length*self.word_embed,self.first_reduce)
        self.word_pos_emb_reduce = LearnedSinPositionalEmbedding(sequence_length,self.first_reduce)
        
        self.reduce_2 = nn.Linear(self.sequence_length*self.first_reduce,self.second_reduce)
        self.word_pos_emb_reduce_2 = LearnedSinPositionalEmbedding(self.sequence_length,self.second_reduce)
        
        self.up_1 = nn.Linear(self.second_reduce, self.up_scale)
        self.up_2 = nn.Linear(self.up_scale,self.vocab_size)
        self.final = nn.Linear(self.sequence_length, 1)
        
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape, "first embedding")
        #add pos emb here
        x = self.word_pos_emb(x)
        #print(x.shape, "after embedding")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "after reshape")
        x = F.relu(self.reduce(x))
        x = self.word_pos_emb_reduce(x)
        #add pos emb here
        #print(x.shape, "after reduce first cross")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape,"second ein")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "another reshape")
        x = F.relu(self.reduce_2(x))
        x = self.word_pos_emb_reduce_2(x)
        #as pos emb here
        #print(x.shape, "second reduce")
        x = F.relu(self.up_1(x))
        #print(x.shape, "up_1")
        x = F.relu(self.up_2(x))
        #print(x.shape,"up_2")
        x = x.transpose(1, 2)
        x = F.relu(self.final(x))
        x = x.squeeze(-1)
        return x
    
    def to(self, device):
        self = super().to(device)
        self.word_pos_emb.positional_input = self.word_pos_emb.positional_input.to(device)
        self.word_pos_emb_reduce.positional_input = self.word_pos_emb_reduce.positional_input.to(device)
        self.word_pos_emb_reduce_2.positional_input = self.word_pos_emb_reduce_2.positional_input.to(device)
        return self   

class Model_1(nn.Module):
    """Learned Sin embeddings, and square memory, some linear layers at the end.  one projection to the vocab pred"""
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                 , linear_dim
                ):
        super(Model_1, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        self.linear_dim = int(linear_dim)
        self.word_pos_emb = LearnedSinPositionalEmbedding(self.sequence_length,self.word_embed)
        
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        self.reduce = nn.Linear(self.sequence_length*self.word_embed,self.word_embed*2)
        self.word_pos_emb_reduce = LearnedSinPositionalEmbedding(sequence_length,self.word_embed*2)
        
        self.reduce_2 = nn.Linear(self.sequence_length*self.word_embed*2,self.word_embed*5)
        self.word_pos_emb_reduce_2 = LearnedSinPositionalEmbedding(sequence_length,self.word_embed*5)
        
        self.down_2 = nn.Linear(self.word_embed*5,self.linear_dim)
        
        self.flat_down = nn.Linear(self.linear_dim*self.sequence_length,self.linear_dim)
        
        self.deep_1 = nn.Linear(self.linear_dim,self.linear_dim)
        self.deep_2 = nn.Linear(self.linear_dim,self.linear_dim)
        self.deep_3 = nn.Linear(self.linear_dim,self.linear_dim)
        
        self.project = nn.Linear(self.linear_dim,vocab_size)
        
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape, "first embedding")
        #add pos emb here
        x = self.word_pos_emb(x)
        #print(x.shape, "after embedding")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "after reshape")
        x = F.relu(self.reduce(x))
        x = self.word_pos_emb_reduce(x)
        #add pos emb here
        #print(x.shape, "after reduce first cross")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape,"second ein")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "another reshape")
        x = F.relu(self.reduce_2(x))
        x = self.word_pos_emb_reduce_2(x)
        #as pos emb here
        #print(x.shape, "second reduce")
        x = F.relu(self.down_2(x))
        #print(x.shape, "final down")
        x = x.reshape(x.shape[0],-1)
        #print(x.shape)
        x = F.relu(self.flat_down(x))
        #print(x.shape)
        x = F.relu(self.deep_1(x))
        x = F.relu(self.deep_2(x))
        x = F.relu(self.deep_3(x))
        x = self.project(x)
        return x
    
    def to(self, device):
        self = super().to(device)
        self.word_pos_emb.positional_input = self.word_pos_emb.positional_input.to(device)
        self.word_pos_emb_reduce.positional_input = self.word_pos_emb_reduce.positional_input.to(device)
        self.word_pos_emb_reduce_2.positional_input = self.word_pos_emb_reduce_2.positional_input.to(device)
        return self
    
class Model_2(nn.Module):
    """Learned Sin embeddings, and square memory, some linear layers at the end.  one projection to the vocab pred"""
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                 , linear_dim
                 , first_reduce = None
                 , second_reduce = None
                 
                ):
        super(Model_2, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        self.linear_dim = int(linear_dim)
        self.word_pos_emb = LearnedSinPositionalEmbedding(self.sequence_length,self.word_embed)
        
        if first_reduce is None:
            self.first_reduce = self.word_embed*2
        else:
            self.first_reduce = int(first_reduce)
        
        if second_reduce is None:
            self.second_reduce = self.word_embed*5
        else:
            self.second_reduce = int(second_reduce)
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        self.reduce = nn.Linear(self.sequence_length*self.word_embed,self.first_reduce)
        self.word_pos_emb_reduce = LearnedSinPositionalEmbedding(sequence_length,self.first_reduce)
        
        self.reduce_2 = nn.Linear(self.sequence_length*self.first_reduce,self.second_reduce)
        self.word_pos_emb_reduce_2 = LearnedSinPositionalEmbedding(sequence_length,self.second_reduce)
        
        self.down_2 = nn.Linear(self.second_reduce,self.linear_dim)
        
        self.flat_down = nn.Linear(self.linear_dim*self.sequence_length,self.linear_dim)
        
        self.deep_1 = nn.Linear(self.linear_dim,self.linear_dim)
        self.deep_2 = nn.Linear(self.linear_dim,self.linear_dim)
        self.deep_3 = nn.Linear(self.linear_dim,self.linear_dim)
        
        self.project_1 = nn.Linear(self.linear_dim,vocab_size)
        self.project_2 = nn.Linear(self.linear_dim,vocab_size)
        self.project_3 = nn.Linear(self.linear_dim,vocab_size)
        
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape, "first embedding")
        #add pos emb here
        x = self.word_pos_emb(x)
        #print(x.shape, "after embedding")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "after reshape")
        x = F.relu(self.reduce(x))
        x = self.word_pos_emb_reduce(x)
        #add pos emb here
        #print(x.shape, "after reduce first cross")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape,"second ein")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "another reshape")
        x = F.relu(self.reduce_2(x))
        x = self.word_pos_emb_reduce_2(x)
        #as pos emb here
        #print(x.shape, "second reduce")
        x = F.relu(self.down_2(x))
        #print(x.shape, "final down")
        x = x.reshape(x.shape[0],-1)
        #print(x.shape)
        x = F.relu(self.flat_down(x))
        #print(x.shape)
        x = F.relu(self.deep_1(x))
        p1 = self.project_1(x)
        #print(p1.shape)
        x = F.relu(self.deep_2(x))
        p2 = self.project_2(x)
        #print(p2.shape)
        x = F.relu(self.deep_3(x))
        p3 = self.project_1(x)
        #print(p3.shape)
        x = p1 + p2 + p3
        #print(x.shape)
        return x
    
    def to(self, device):
        self = super().to(device)
        self.word_pos_emb.positional_input = self.word_pos_emb.positional_input.to(device)
        self.word_pos_emb_reduce.positional_input = self.word_pos_emb_reduce.positional_input.to(device)
        self.word_pos_emb_reduce_2.positional_input = self.word_pos_emb_reduce_2.positional_input.to(device)
        return self
    
class Model_3(nn.Module):
    """Learned Sin embeddings, and square memory, some linear layers at the end.  several projections from the same latent"""
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                 , linear_dim
                ):
        super(Model_3, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        self.linear_dim = int(linear_dim)
        self.word_pos_emb = LearnedSinPositionalEmbedding(self.sequence_length,self.word_embed)
        
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        self.reduce = nn.Linear(self.sequence_length*self.word_embed,self.word_embed*2)
        self.word_pos_emb_reduce = LearnedSinPositionalEmbedding(sequence_length,self.word_embed*2)
        
        self.reduce_2 = nn.Linear(self.sequence_length*self.word_embed*2,self.word_embed*5)
        self.word_pos_emb_reduce_2 = LearnedSinPositionalEmbedding(sequence_length,self.word_embed*5)
        
        self.down_2 = nn.Linear(self.word_embed*5,self.linear_dim)
        
        self.flat_down = nn.Linear(self.linear_dim*self.sequence_length,self.linear_dim)

        
        self.project_1 = nn.Linear(self.linear_dim,vocab_size)
        self.project_2 = nn.Linear(self.linear_dim,vocab_size)
        self.project_3 = nn.Linear(self.linear_dim,vocab_size)
        
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape, "first embedding")
        #add pos emb here
        x = self.word_pos_emb(x)
        #print(x.shape, "after embedding")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "after reshape")
        x = F.relu(self.reduce(x))
        x = self.word_pos_emb_reduce(x)
        #add pos emb here
        #print(x.shape, "after reduce first cross")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape,"second ein")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "another reshape")
        x = F.relu(self.reduce_2(x))
        x = self.word_pos_emb_reduce_2(x)
        #as pos emb here
        #print(x.shape, "second reduce")
        x = F.relu(self.down_2(x))
        #print(x.shape, "final down")
        x = x.reshape(x.shape[0],-1)
        #print(x.shape)
        x = F.relu(self.flat_down(x))
        
        p1 = self.project_1(x)
        p2 = self.project_2(x)
        p3 = self.project_3(x)
        x = p1 + p2 + p3
        
        return x
    
    def to(self, device):
        self = super().to(device)
        self.word_pos_emb.positional_input = self.word_pos_emb.positional_input.to(device)
        self.word_pos_emb_reduce.positional_input = self.word_pos_emb_reduce.positional_input.to(device)
        self.word_pos_emb_reduce_2.positional_input = self.word_pos_emb_reduce_2.positional_input.to(device)
        return self
    
class Model_4(nn.Module):
    """Learned Sin embeddings, and square memory, some linear layers at the end. a memory project and a sentence project"""
    def __init__(self
                 , vocab_size
                 , sequence_length
                 , word_embed
                 , linear_dim
                ):
        super(Model_4, self).__init__()
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.word_embed = int(word_embed)
        self.linear_dim = int(linear_dim)
        self.word_pos_emb = LearnedSinPositionalEmbedding(self.sequence_length,self.word_embed)
        self.mem_dim = 500
        
        self.word_embedding = nn.Linear(self.vocab_size,self.word_embed) 
        self.reduce = nn.Linear(self.sequence_length*self.word_embed,self.word_embed*2)
        self.word_pos_emb_reduce = LearnedSinPositionalEmbedding(sequence_length,self.word_embed*2)
        
        self.reduce_2 = nn.Linear(self.sequence_length*self.word_embed*2,self.word_embed*5)
        self.word_pos_emb_reduce_2 = LearnedSinPositionalEmbedding(sequence_length,self.word_embed*5)
        
        self.down_2 = nn.Linear(self.word_embed*5,self.linear_dim)
        
        self.flat_down = nn.Linear(self.linear_dim*self.sequence_length,self.linear_dim)
        
        self.mem_adjust = nn.Linear(self.linear_dim,self.linear_dim)
        
        self.mem = SquareMemory(self.linear_dim, self.mem_dim)
        
        self.x_project = nn.Linear(self.linear_dim,vocab_size)
        self.mem_project = nn.Linear(self.mem_dim,vocab_size)
        
        
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.word_embedding(x)) #sentence of word embeddings.  
        #print(x.shape, "first embedding")
        #add pos emb here
        x = self.word_pos_emb(x)
        #print(x.shape, "after embedding")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape, "after einsum")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "after reshape")
        x = F.relu(self.reduce(x))
        x = self.word_pos_emb_reduce(x)
        #add pos emb here
        #print(x.shape, "after reduce first cross")
        x = torch.einsum('bij,bkm->bikj', x, x)
        #print(x.shape,"second ein")
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #print(x.shape, "another reshape")
        x = F.relu(self.reduce_2(x))
        x = self.word_pos_emb_reduce_2(x)
        #as pos emb here
        #print(x.shape, "second reduce")
        x = F.relu(self.down_2(x))
        #print(x.shape, "final down")
        x = x.reshape(x.shape[0],-1)
        #print(x.shape)
        x = F.relu(self.flat_down(x))
        
        m_x = F.relu(self.mem_adjust(x))
        mem = self.mem(m_x)
        #print(mem.shape)
        p1 = self.x_project(x)
        p2 = self.mem_project(mem)
        
        x = p1 + p2
        return x
    
    def to(self, device):
        self = super().to(device)
        self.word_pos_emb.positional_input = self.word_pos_emb.positional_input.to(device)
        self.word_pos_emb_reduce.positional_input = self.word_pos_emb_reduce.positional_input.to(device)
        self.word_pos_emb_reduce_2.positional_input = self.word_pos_emb_reduce_2.positional_input.to(device)
        return self
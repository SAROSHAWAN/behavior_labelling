import torch
import torch.nn as nn

#TODO:sarosh pls implement
class WindowGRU(nn.Module):
    def __init__(self, ):#implement the gru
        #input_dim=768 (sbert encoding dim), hidden_dim=256, output_dim=6 (bart label 1x6)
        return "unimplemented"
        
        
    def forward(self, x): #implement how the forward pass looks like
        #i will implement helper funct to create sequence chunk for each pass
        #req for pass input : [8, 100, 768]
        # output: [8,100, 6] 
        
        return "unimplemented"
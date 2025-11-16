import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''

In this file you will write the model definition for a feedforward neural network. 

Please only complete the model definition and do not include any training code.

The model should be a feedforward neural network, that accepts 784 inputs (each image is 28x28, and is flattened for input to the network)
and the output size is 10. Whether you need to normalize outputs using softmax depends on your choice of loss function.

PyTorch documentation is available at https://pytorch.org/docs/stable/index.html, and will specify whether a given loss funciton 
requires normalized outputs or not.

'''

class FF_Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
        

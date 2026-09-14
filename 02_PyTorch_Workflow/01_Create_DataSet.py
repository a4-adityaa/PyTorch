# Let's create some knwon data

import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# creating known paramenter
weight=0.3
bias=0.7

# creating data

start=0
ending=1
step=0.02

x= torch.arange(start, ending, step).unsqueeze(dim=1)
y= weight*x + bias
print(x[:10],y[:10]) 
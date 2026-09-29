import torch
from torch import nn
import matplotlib.pyplot as plt

a= torch.arange(-10,10,1, dtype=(torch.float32))

# plt.plot(a)

def relu(x):
    return torch.maximum(torch.tensor(0),x)

# plt.plot(relu(a))

def sigmoid(x):
    return 1/(1+torch.exp(-x))

plt.plot(sigmoid(a))
plt.show()
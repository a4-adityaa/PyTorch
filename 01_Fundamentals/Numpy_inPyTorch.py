# Numpy is a numerical computing library, stores data in form of arrays and shares same memory;

import torch
import numpy as np

array= np.arange(1.,11.)
#print(array)

# to convert numpy.array into pytorch.tensor 

tensor= torch.from_numpy(array).type(torch.float32) # we can also use (torch.tensor(array).to(torch.float32)) and 
#print(f"the tensored array: {tensor} and it's datatype is {tensor.dtype}")

newArray = tensor.numpy() #np.array(tensor) -> can give conversion error but programs runs
print(f"tensor to array: {newArray}")
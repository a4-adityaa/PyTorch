# Tensor.stack is used to combime multile stack in one new diamension

import torch as tr

x= tr.tensor([2,5,8])
y= tr.tensor([7,3,6])

stacked_tensor= tr.stack((x,y))

print(f"the new tensor is: {stacked_tensor.tolist()} and it's size is: {stacked_tensor.shape}")
# indexing is used to find the elemnts in tensors

import torch as tr

x= tr.tensor([[[2,5,8],[7,3,6],[8,10,1]]])

print(x[0].tolist()) # outer diamension
print(x[0][0].tolist()) # inner diamension i.e rows and columns
print(x[0][0][0].tolist()) 

print(x[0][2][1].tolist())
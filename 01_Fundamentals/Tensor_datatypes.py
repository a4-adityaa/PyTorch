# Datatypes in pytorch

import torch as tr

x = tr.tensor([3,4,6], 
              dtype=tr.float32,
              device= None)
print(x)
print(x.dtype)
print(x.device)

y= tr.tensor([5,6,9],
             dtype=tr.int32)
print(y)
print(y.dtype)

# Multiplication of two data types
z= x*y
print(z)
print(z.dtype)
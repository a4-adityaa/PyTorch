# Reshape is used to add extra diamension to the tensor

import torch as tr

x= tr.arange(start=1, end=11)

print(x.tolist())
# print(x.size())

reshape_X = x.reshape(1,-1) 
''' -1 calculates the number of diamension already persists and 1 is new diamnesion to be added
    Total elements divisible hone chahiye fixed dimension se.'''
print(f"the new diamension is: {reshape_X}")

view= x.view(2,-1)
# view mean: same tensor data ko different shape mein dekhna (same as reshape but it share same meory as x so we can change value too using indexing)
print(f"the view of the tensor is: {view}")
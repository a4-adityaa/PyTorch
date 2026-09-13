# postional mean and max also known as argmin and argmax  are used to find the index of the number

import torch as tr

x= tr.tensor([5,10,3,15,4,6,1])

print(f"the postion of min number is :{tr.argmin(x)}")
print(f"the postion of max number is :{tr.argmax(x)}")

# to find postion in 2D array
y= tr.tensor([[3,5,8],[7,3,2]])

print(f"the min at: {tr.unravel_index(tr.argmin(y), y.shape)}") # returns tensor(1), tensor(2))
print(f"the max at: {tr.unravel_index(tr.argmax(y), y.shape)}") # returns tensor(0), tensor(2))
# Generate zeros and ones

import torch as tr

#  Zero tensor
x= tr.zeros(3,4) # also cam tr.zeros(size=(3,3))
print(x)

# ones tensor

y = tr.ones(size=(3,4))
print(y)
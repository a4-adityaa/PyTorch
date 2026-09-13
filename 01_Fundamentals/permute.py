# permute
'''its like transpose, the rows convert into columns '''

import torch as tr

x= tr.tensor([[5,2,3],
              [4,8,1]])

permuted= x.permute(0,1)

print(f"the diamension before permute: {x.shape}")
print(f"the diamension after permute: {permuted.shape}")
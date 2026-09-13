# Unsqueeze and unsqueeze;
'''
    Unsqueeze add dimension while squeeze remove diamension.
    unsqueeze(0) -> add dimension in begining 
    unsqueeze(1) -> add dimension in second position
    squeeze() -> remove all the diamension with size 1
'''
import torch as tr

x= tr.tensor([2,5,6])
y= tr.tensor([[4],[5],[6]])

# Unsqueeze
unsq0 = x.unsqueeze(0)
print(f"the size before unsqueeze(0) is: {x.shape}")
print(f"the size after unsqueeze(0) is: {unsq0.shape}")

unsq1 = x.unsqueeze(1)
print(f"the size before unsqueeze(1) is: {x.shape}")
print(f"the size after unsqueeze(1) is: {unsq1.shape}")

# Squeeze
sq0= y.squeeze(0)
print(f"size before squeeze(0): {y.shape}")
print(f"size before squeeze(0): {sq0.shape}")

sq1 = y.squeeze(1)
print(f"size before squeeze(1): {y.shape}")
print(f"size before squeeze(1): {sq1.shape}")
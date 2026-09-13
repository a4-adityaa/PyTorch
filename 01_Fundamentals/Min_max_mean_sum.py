# let's perform some aggrete function intensors

import torch as tr

x= tr.arange(10,101,10) # 10 for staring, 101 for ending and 10 for steps

print(f"the data we have: {x.tolist()}")

# min
min= tr.min(x) 
or_min=x.min()

print(f"the min is: {min} or {or_min}")

# max
max= tr.max(x)
or_max= x.max()
print(f"the max is: {max} or {or_max}")

# mean
mean= tr.mean(x.float())  # float is necessary because mean can gives result in decimal form
or_mean= x.float().mean()
print(f"the mean is : {mean} or {or_mean}")

# sum
sum= x.sum() 
or_sum= tr.sum(x)
print(f"the sum is: {sum} or {or_sum}")
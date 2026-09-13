'''
    Reprducebility is used to control on the automatically generated number using .rand or .radians
    from same seed we get same random sequence 
'''
import torch as tr

tr.manual_seed(10)

x= tr.rand(5)
y= tr.rand(6)

print(x.tolist())
print(y.tolist())
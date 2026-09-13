# Random tensors are used to randomaly generate numbers, in DL it's mostly used for intilize weights of Neural netwoks
import torch as tr

x=tr.rand(3,3)

print(x)
print(x.shape)

k=tr.rand(2)
print(k)
print(k.shape)

image_tensor=tr.rand(size=(3,234,250))
# print(image_tensor)
print(image_tensor.size())
print(image_tensor.ndim)
print(image_tensor.shape)

# Can set limits else it generate number btwm 0-1
y=tr.randint(
    high=10,
    low=1,
    size=(3,3) 
)
print(y)
print(y.shape)
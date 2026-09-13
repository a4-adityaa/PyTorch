import torch as tr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# print(torch.__version__)

# Scalar Tensor
scalar = tr.tensor(7)

print(scalar) # output : tensor(7)
print(scalar.item())
print(scalar.ndim) # output = 0 -> shows the number of diamensions

# Vector Tensor
vector = tr.tensor([7,7])
print(vector) # Output : tenosr([7,7])
print(vector.tolist())
print(vector.ndim) # output: 1 -> it's a single list not a 2D matrix
print(vector.shape) # output: 2 -> size of every diamension

# Matrix Tensor
matrix = tr.tensor([[5,6,4],
                    [2,3,7]])
print(matrix)
print(matrix.tolist())
print(matrix.shape) # output:[2,3] -> size of diamnesion i.e 2 rows and 3 columns
print(matrix.ndim) # Output: 2 -> Matrix is a 2D 
print(matrix[1]) #Output: [2,3,7] -> works as index like an array

# Tensor
_3Dmatrix = tr.tensor([[[5,3,2],[2,7,1],[1,0,6]]])
print(_3Dmatrix)
print(_3Dmatrix.ndim)
print(_3Dmatrix.shape) # Output:[1,3,3] ->[layers, rows,columns]
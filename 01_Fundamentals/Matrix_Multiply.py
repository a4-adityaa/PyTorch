# Multiplication of matrix

import torch as tr

x= tr.tensor([[2,3,4],
              [3,8,1]])
y= tr.tensor([[[5,2],
               [3,1],
               [7,3]]])
z= tr.tensor([[5,2,3],
              [1,7,3]])

# Element multiplication
Ele_multipliaction = x * x # Multiplies elemnt with each other within same matrix  i.e 2*3 + 3*8 + 4*1
print(f"Element multiplication: {Ele_multipliaction.tolist()}")

# Metric multiplication (matmul) or we can '@' as a operator to multiply;
matric_mul= tr.matmul(x,y)
print(f"Using matmul: {matric_mul.tolist()}")


'''Matrix rule 
1.Matrix multiplication is only possible when the number of columns in the first matrix is equal to the number of rows in the second matrix.
2. We can't multiply metrics like (2,3)*(2,3)

 so to overcome this, we use transpose method which covert the metrics rows to column and column to row;
'''

Transpose_Matrix= tr.matmul(x,z.T) # we use.T or Transpose to use tranpose (torch.mm can also be used at torch.matmul)
print(f"After transposing the matrices: {Transpose_Matrix.tolist()}")
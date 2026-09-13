# We can perform operations on tensors i.e Addition, subtractioon, multipliaction etc...

import torch as tr

x= tr.tensor([2,5,6])

# additiom
Addition= x+10
print(f"The addition is: {Addition.tolist()}")

# Multiplication
Multiplication= x*10
print(f"the multiplication is: {Multiplication.tolist()}")
# Multiply=tr.multiply(x,10) # Also we can use torch command

# Subtraction
Subtraction= x-10
print(f"The subtraction is: {Subtraction.tolist()}")

# Division
Division= x/2
print(f"The division is: {Division.tolist()}")

# Square
Square= x**2
print(f"The square is: {Square.tolist()}")
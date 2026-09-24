import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import torch

# create sample

n_sample=1000

x,y = make_circles(n_sample,
                   noise=0.03,
                   random_state=42)

# print(f"First five data of x:\n {x[:5]}")
# print(f"First five data of y:\n {y[:5]}")

# make dataframe of circle data
circles= pd.DataFrame({"x1": x[:,0], # Dataframe is used to visulaise the data in tabular form (a pandas fctn)
                      "x2": x[:,1],
                      "label": y})

# print(circles.head(10)) # cicle.head() is used to define the no of data we want

# print(circles.label.value_counts()) # counts diffeerent lables 

# visulize with a plot
plt.scatter(x=x[:,0],
            y=x[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)

# plt.show()

# View the first example of features and labels
X_sample = x[0]
y_sample = y[0]
# print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
# print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

# noe convert data to tensor
x= torch.from_numpy(x).type(torch.float32)
y= torch.from_numpy(y).type(torch.float32)

# print(f" datatypes of x: {x.dtype}")
# print(f" datatypes of y: {y.dtype}")
# print(f"first 5 data are : {x[:5]}")
# print(f"first 5 data are : {y[:5]}")

# now split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,
                                                   y,
                                                   test_size=0.2, # 20% will be testing and 80% will be traing data
                                                   random_state=42) # random state is same as torch.manual_seed -> diffrent for  sckit learn

print(len(x_train), len(y_train), len(x_test), len(y_test))
# now we split our data set  

import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# creating known paramenter
weight=0.3
bias=0.7

# creating data

start=0
ending=1
step=0.02

x= torch.arange(start, ending, step).unsqueeze(dim=1)
y= weight*x + bias

train_split = int(0.8 * len(x))
x_train, y_train= x[:train_split], y[:train_split]
x_text, y_test = x[train_split:], y[train_split:]

# print(x_train, y_train, x_text, y_test)

# now create function to visulaize it

def plot_predictions(train_data=x_train,  # we are giving defalut parameters if no parameter is passed  during fctn call they will execute
                     train_labels=y_train,
                     test_data=x_text,
                     test_labels=y_test,
                     predictions=None):

    plt.figure(figsize=(10,7)) # used to define the size of graph

    # plot training data in blue color
    plt.scatter(train_data, train_labels, c="b", s=4, label="training data") 
    # c is color, s= plot size(defines the distace btwn dots), lable is used to name the set
    
    # plot testing data
    plt.scatter(test_data, test_labels, c="g", s=4, label="text data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="predictions")

    plt.legend(fontsize=14) # used to define the size of text labels
    plt.show() # show the graphs

plot_predictions()
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# create data
weight=0.7
bias=0.

start=0
end=1
step=0.02

x = torch.arange(start,end,step)
y= weight*x + bias
# print(f"Data created: {x,y}")
# print(x[:10],y[:10])

# split data
train_split= int(0.8*len(x))
x_train, y_train= x[:train_split], y[:train_split]
x_test, y_test= x[train_split:], y[train_split:]

# print(len(x_train), len(y_train), len(x_test),len(y_test))

# now plot data to visualize
def plot_predictions(train_data=x_train,  
                     train_labels=y_train,
                     test_data=x_test,
                     test_labels=y_test,
                     predictions=None):

    plt.figure(figsize=(10,7)) 

    # plot training data in blue color
    plt.scatter(train_data, train_labels, c="b", s=4, label="training data") 
    
    # plot testing data
    plt.scatter(test_data, test_labels, c="g", s=4, label="test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="predictions")

    plt.legend(fontsize=14)
    # plt.show() # show the graphs

# plot_predictions()
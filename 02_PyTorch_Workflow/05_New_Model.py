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

# now we make a model using linear.nn

class LinearRegressionModelv2(nn.Module):
    def __init__(self):
        super().__init__()

        # now we use nn.linear instead of defining weight and bias;
        self.linear_layer= nn.Linear(in_features=1, out_features=1) # in_features is the no of features we have in our data
                                                                      # out_features is the no of features we want to predict

        def forward(self, x: torch.tensor) -> torch.tensor:
            return self.linear_layer(x)

# set manual seed;
torch.manual_seed(42)
model_1 = LinearRegressionModelv2()
# print(model_1.state_dict())

loss_fn= nn.L1Loss() # mean absolute error

optimizer= torch.optim.SGD(params=model_1.parameters(), lr=0.01)

epochs=100

for epoch in range(epochs):
    # set the model to training mode
    model_1.train() 

    # forward pass
    y_pred= x_train(model_1)

    # calculate loss
    loss_fn= nn.L1Loss() # mean absolute error
    loss= loss_fn(y_pred,x_test)

    # set optimizer
    optimizer.zero_grad()
    # loss backward
    loss.backward()
    # optimizer step
    optimizer.step()

    # testing
    model_1.eval() # set the model to evaluation mode

    with torch.inference_mode:
        test_pred= model_1(x_test)

    # calculate loss
    test_loss= loss_fn(y_pred, x_test.dtype(torch.float32))
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss}")
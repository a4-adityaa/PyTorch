# now we split our data set  

import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
from pathlib import Path

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
x_test, y_test = x[train_split:], y[train_split:]

# print(x_train, y_train, x_text, y_test)

# now create function to visulaize it

def plot_predictions(train_data=x_train,  # we are giving defalut parameters if no parameter is passed  during fctn call they will execute
                     train_labels=y_train,
                     test_data=x_test,
                     test_labels=y_test,
                     predictions=None):

    plt.figure(figsize=(10,7)) # used to define the size of graph

    # plot training data in blue color
    plt.scatter(train_data, train_labels, c="b", s=4, label="training data") 
    # c is color, s= plot size(defines the distace btwn dots), lable is used to name the set
    
    # plot testing data
    plt.scatter(test_data, test_labels, c="g", s=4, label="test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="predictions")

    plt.legend(fontsize=14) # used to define the size of text labels
    # plt.show() # show the graphs

# plot_predictions()

class linearregressionModel(nn.Module): #nn.module cover all the alogorithm
    def __init__(self):
        super().__init__() # super is used to access the property of class i.e nn.module

        self.weights= nn.Parameter(torch.randn(1, dtype=float, requires_grad=True)) # start with random weight and get adjusted  
                                                                                    # grad is used to track the value for updation

        self.bias= nn.Parameter(torch.randn(1, dtype=float, requires_grad=True)) # start with random bias and get adjusted

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.weights * x + self.bias

# cgecks the content of model:
# set manual seed as data are randomaly intilised;
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = linearregressionModel()

# print(list(model_0.parameters())) # gives list values

# print(model_0.state_dict()) # gives disctionary values

# create prdediction using torch.inference_mode

with torch.inference_mode():
    y_preds = model_0(x_test)

# plot_predictions(predictions=y_preds)
# plt.show()

# now set loss function
loss_fn = nn.L1Loss() # MAE is same as L1 loss

optimizer= torch.optim.SGD(params=model_0.parameters(), # parameneter of target model to train
                           lr=0.01) # learning rate


# Train model;
epochs=1000

# for keep track of data
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model_0.train() # put model in training mode

    y_pred = model_0(x_train) # forward pass

    loss = loss_fn(y_pred, y_train) # calculate loss
    # print(f"loss: {loss}")

    optimizer.zero_grad() # zero the gradients

    loss.backward() # backward pass

    optimizer.step() # update the parameters

    model_0.eval() # put model in evaluation mode

    with torch.inference_mode():
        test_pred = model_0(x_test)
        test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

        if epoch %10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            # print(f"epoch: {epoch} | loss: {loss} | test_loss: {test_loss}")

# print(model_0.state_dict())
# plot_predictions(predictions=test_pred)
# plt.show()

# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();
# plt.show()

# Saving a PyTorch model
# 1. create model directory
MODEL_PATH= Path("Models") 
MODEL_PATH.mkdir(parents=True,exist_ok= True)

# 2. Create model save path
MODEL_NAME="Model_0.pth"
MODEL_SAVE_PATH= MODEL_PATH / MODEL_NAME

# 3. Save model's state dict
print(f"Saving a model to: {MODEL_SAVE_PATH}")

torch.save(                  # saving start from here
    obj=model_0.state_dict(), # kya store kr rhe i.e weights and bias in form of dictionay
    f=MODEL_SAVE_PATH # kis jagah save krna hai
)
# print(MODEL_SAVE_PATH.exists()) # checks whaeter the model exits or not
" Use sequential model to predict linear data "
import torch
from torch import nn
import matplotlib.pyplot as plt
from helper_functions import plot_predictions

# now let's create some data;

weight=0.7
bias = 0.3
st =0
end =1
step =0.01
x_regression =torch.arange(st, end, step).unsqueeze(dim=1)
y_regression = weight* x_regression + bias

# print(x_regression[:5],y_regression[:5])

# Now we will split data
train_split = int(0.8 * len(x_regression))
x_train_regression, y_train_regression = x_regression[:train_split], y_regression[:train_split]
x_test_regression, y_test_regression = x_regression[train_split:], y_regression[train_split:]

# print(len(x_train),len(y_train),len(x_test),len(y_test)) -> prints the lengths

" Plot predictions to visulize "

plot_predictions(train_data=x_train_regression,
                 train_labels=y_train_regression,
                 test_data=x_test_regression,
                 test_labels=y_test_regression)

# plt.show()

# Creating a model
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
)

# print(model_2)

loss_fn = nn.L1Loss()
optimizer= torch.optim.SGD(model_2.parameters(),
                           lr=0.1)

" Now we will train the model"
torch.manual_seed(42)

epochs=500
for epoch in range(epochs):
    model_2.train()

    # forward pass
    y_preds= model_2(x_train_regression)
    loss = loss_fn(y_preds, y_train_regression)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # testing
    model_2.eval()
    with torch.inference_mode():
        test_preds = model_2(x_test_regression)
        test_loss = loss_fn(test_preds, y_test_regression)

    # if epoch % 100 == 0: 
    #     print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")

# Turn on evaluation mode
model_2.eval()

# Make predictions (inference)
with torch.inference_mode():
    y_preds = model_2(x_test_regression)

# Plot data and predictions with data on the CPU (matplotlib can't handle data on the GPU)
# (try removing .cpu() from one of the below and see what happens)
plot_predictions(train_data=x_train_regression,
                 train_labels=y_train_regression,
                 test_data=x_test_regression,
                 test_labels=y_test_regression,
                 predictions=y_preds);

plt.show()
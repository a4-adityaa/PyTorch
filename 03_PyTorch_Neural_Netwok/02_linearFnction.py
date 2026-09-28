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
x_regression =torch.arange(st, end, step)
y_regression = weight* x_regression + bias

# print(x_regression[:5],y_regression[:5])

# Now we will split data
train_split = int(0.8 * len(x_regression))
x_train, y_train = x_regression[:train_split], y_regression[:train_split]
x_test, y_test = x_regression[train_split:], y_regression[train_split:]

# print(len(x_train),len(y_train),len(x_test),len(y_test)) -> prints the lengths

" Plot predictions to visulize "

plot_predictions(train_data=x_train,
                 train_labels=y_train,
                 test_data=x_test,
                 test_labels=y_test)

# plt.show()

model_2 = nn.Sequential(
    nn.Linear(in_features=2, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
)

print(model_2)
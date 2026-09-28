from torch import nn
import torch
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

n_sample= 1000

X,y= make_circles(n_sample,
                noise=0.03,
                random_state=42,)

plt.scatter(X[:,0], X[:,1], c=y, cmap= plt.cm.RdYlBu)
# plt.show()

" perform train and test split "
from sklearn.model_selection import train_test_split

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2,
                                                    random_state=42
)

# print(X_train[:5], y_train[:5])
" Let's create a model "

class CircularModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1= nn.Linear(in_features=2, out_features=10)
        self.layer_2= nn.Linear(in_features=10, out_features=10)
        self.layer_3= nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # -> add in ReLU activation Function (no need to add nn.sigmoid if we are using it.)

    def forward(self, X):
        return self.layer_3(self.ReLU(self.layer_2(self.ReLU(self.layer_1(X)))))

model_3= CircularModelV2()
print(model_3)
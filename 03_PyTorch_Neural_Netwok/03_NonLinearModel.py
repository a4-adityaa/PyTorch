from torch import nn
import torch
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from helper_functions import plot_decision_boundary

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
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(X)))))

model_3= CircularModelV2()
# print(model_3)

" Setup a loss_fn, accuracy and optimizer"

loss_fn= nn.BCEWithLogitsLoss()
optimizer= torch.optim.SGD(params=model_3.parameters(), lr=0.1)
def accuracy_fn(y_true, y_preds):
    correct= torch.eq(y_true, y_preds).sum().item()
    acc= (correct/len(y_preds))*100
    return acc


# Fit the model
torch.manual_seed(42)
epochs = 1000

# Put all data on target device
X_train, y_train = X_train, y_train
X_test, y_test = X_test, y_test

for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=y_train, 
                      y_preds=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_3.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_3(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
      # 2. Calculate loss and accuracy
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_true=y_test,
                             y_preds=test_pred)

    # Print out what's happening
    # if epoch % 100 == 0:
    #     print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

# Make predictions
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()
y_preds[:10], y[:10] # want preds in same format as truth labels

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_train) # model_1 = no non-linearity
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test) # model_3 = has non-linearity

plt.show()
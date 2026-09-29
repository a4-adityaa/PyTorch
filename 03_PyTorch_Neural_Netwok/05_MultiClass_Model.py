import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import nn
from helper_functions import plot_decision_boundary

" Here we are gonna create a multiclass Model "
NUM_CLASSES =4
NUM_FEATURES =2
RANDOM_SEED= 42

X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers= NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

# Now convert our data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.long)

# Now split data into training and testing
X_blob_train, X_blob_test, y_blob_train, y_blob_test= train_test_split(X_blob, y_blob,
                                                                       test_size=0.2,
                                                                       random_state=RANDOM_SEED)

plt.figure(figsize=(10,7))
plt.scatter(X_blob[:,0], X_blob[:,1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()

"  Now Create a model "

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_unit):
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_unit),
            # nn.ReLU(), -> does our model requires nn.ReLU
            nn.Linear(in_features=hidden_unit, out_features=hidden_unit),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_unit, out_features=output_features)
        )
    # Forward pass
    def forward(self,X):
        return self.linear_layer_stack(X)

# Now create instance of our model
model_4= BlobModel(input_features=2,output_features=4, hidden_unit=8)
# print(model_4)

" Now create a loss_fn and Optimizer alongwith accuracy "
loss_fn= nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(model_4.parameters(),
                           lr=0.1)

def accuracy_fn(y_true, y_preds):
    correct= torch.eq(y_true, y_preds).sum().item()
    acc= (correct/len(y_preds))*100
    return acc

" Now we will create a Training and testing loop "

torch.manual_seed(42)
epochs=100

for epoch in range(epochs):
    model_4.train()

    # Forward pass
    y_logits= model_4(X_blob_train) # model output raw logits
    y_preds= torch.softmax(y_logits, dim=1).argmax(dim=1) # go from logits -> prediction probablities -> prediction lables

    loss= loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                           y_preds=y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Testing
    model_4.eval()
    with torch.inference_mode():
        test_logits= model_4(X_blob_test)
        test_preds= torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss= loss_fn(test_logits, y_blob_test)
        test_acc= accuracy_fn(y_true=y_blob_test,
                              y_preds=test_preds)

    # Print out what's happening
    # if epoch % 10 == 0:
    #     print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

# Make predictions
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

# View the first 10 predictions
# print(y_logits[:10])

" Now let's plot our data "

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)

plt.show()
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
y_blob = torch.from_numpy(y_blob).type(torch.float)

# Now split data into training and testing
X_blob_train, X_blob_test, y_blob_train, y_blob_test= train_test_split(X_blob, y_blob,
                                                                       test_size=0.2,
                                                                       random_state=RANDOM_SEED)

plt.figure(figsize=(10,7))
plt.scatter(X_blob[:,0], X_blob[:,1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()
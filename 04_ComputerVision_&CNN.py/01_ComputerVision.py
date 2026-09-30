import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

" We are going to use FashionMNIST for geeting data "

train_data= datasets.FashionMNIST(
    root= "Data", # -> where we want data
    train= True, # want traing data or not (gives training data)
    download= True, # want to download
    transform= torchvision.transforms.ToTensor(), # hwo do we want to tranform data
    target_transform= None # how we want to transform lables/targets
)

test_data= datasets.FashionMNIST(
    root="Data",
    train=False,
    download= True,
    transform= ToTensor(), # can be also written like this
    target_transform= None
)
 
# print(len(train_data), len(test_data))

# see the first training data
image, label= train_data[0]
# print(image,label)
# print(image.shape) # -> shape of our data
# print(len(train_data),len(train_data.targets), len(test_data),len(test_data.targets)) # -> Numbers of samples

 # see classes
class_name = train_data.classes
# print(class_name) # -> gives the total classes

class_idx= train_data.class_to_idx
# print(class_idx) # -> gives dictionary of classes and index

# Now Visulize
# image, label = train_data[0]
# print(f"Image shape: {image.shape}")
# plt.imshow(image.squeeze(), cmap="grey") # image shape is [1, 28, 28] (colour channels, height, width)
# plt.title(class_name[label]);
# plt.axis(False)
# plt.show()

# Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
'''
for i in range(1, rows * cols + 1): # loops start from 1 and goes to (4*4+1) i.e 17 can say 16 images 
    random_idx = torch.randint(0, len(train_data), size=[1]).item() # selects the random data from the dataset and .item is used to show data in integer form
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i) # this plots the data on every index of i
    plt.imshow(img.squeeze(), cmap="gray") # removes extra diamensions and cmap="gery" -> refers to color 
    plt.title(class_name[label]) # show titles
    plt.axis(False); # hides tha axis
plt.show()
'''

from torch.utils.data import DataLoader
BATCH_SIZE=32

train_dataloader= DataLoader(train_data, # the data we want to iterate
                             batch_size=BATCH_SIZE, # no of elemnts at once we want to give to Model
                             shuffle= True) # shuffle data ever epochs

test_dataloader= DataLoader(test_data,
                            batch_size=BATCH_SIZE,
                            shuffle=False) # do not need to necceserily shuffle data

# let's check the number of dataloader
# print(f"the no of train dataloader: {len(train_dataloader)} in batches of: {BATCH_SIZE}")
# print(f"the no of test dataloader: {len(test_dataloader)} in batches of: {BATCH_SIZE}")

# Checks whats inside training dataloader
train_features_batch,train_label_batch =next(iter(train_dataloader))
# print(train_features_batch.shape, train_label_batch.shape)

# now visulize the train_features_batch dataloader

for i in range(1, rows*cols+1):
    random_idx= torch.randint(0, len(train_label_batch), size=[1]).item()
    img, label= train_features_batch[random_idx], train_label_batch[random_idx]
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_name[label])
    plt.axis(False)
plt.show()

'''  Now let's create a model '''

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_unit: int, output_shape: int):
        super().__init__()
        self.layer_stack= nn.Sequential(
            nn.Flatten(), # -> flatten the input data
            nn.Linear(in_features=input_shape, out_features=hidden_unit), # -> first linear layer
            nn.Linear(in_features=hidden_unit, out_features=hidden_unit), # -> second linear layer
            nn.Linear(in_features=hidden_unit, out_features=output_shape) # -> third linear layer
        )

    # forward pass
    def forward(self, x):
        self.layer_stack(x)
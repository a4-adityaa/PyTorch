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

''' # see classes
class_name = train_data.classes
print(class_name) # -> gives the total classes

class_idx= train_data.class_to_idx
print(class_idx) # -> gives dictionary of classes and index

'''
# Now Visulize
image, label = train_data[0]
# print(f"Image shape: {image.shape}")
plt.imshow(image.squeeze()) # image shape is [1, 28, 28] (colour channels, height, width)
plt.title(label);
plt.show()
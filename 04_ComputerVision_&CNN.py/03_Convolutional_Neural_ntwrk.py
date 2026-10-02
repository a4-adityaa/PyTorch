''' Gonna build a model_2 with the help of CNN (Convolutional Neural Networks) '''
"Let's import dependencies and create some data "

import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# data
train_data= datasets.FashionMNIST(
    root="Data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data= datasets.FashionMNIST(
    root="Data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

image,label= train_data[0]
# print(image, label)
# print(len(train_data), len(train_data.targets), len(test_data), len(test_data.targets))

"prepare data loader "

from torch.utils.data import DataLoader
BATCH_SIZE=32

train_dataloader= DataLoader(train_data,
                             batch_size= BATCH_SIZE,
                             shuffle= True)

test_dataloader= DataLoader(test_data,
                            batch_size= BATCH_SIZE,
                            shuffle=False)

class_name = train_data.classes
class_idx= train_data.class_to_idx

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

" Now create a model "

class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_unit: int, output_shape: int):
        super().__init__()
        self.block_1= nn.Sequential(
            nn.Conv2d(in_channels= input_shape,  # Conv2d ek learnable filter operation hai jo training ke through useful spatial patterns/features detect karna seekhta hai.
                      out_channels=hidden_unit,
                      kernel_size=3, # Kitna area ek baar mein dekh raha hoon?
                      stride=1,      # Kitne pixels jump karke next jagah jaa raha hoon?
                      padding=1),    # Border ke bahar kitni extra space add kar raha hoon?   refrences CNN.explainer
            nn.ReLU(),              # adds non~linearity
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, # reduces the size of spatial map and retains the important and strongest feature
                         stride=2),
        )
        self.block_2= nn.Sequential(
            nn.Conv2d(in_channels=hidden_unit,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_unit, hidden_unit,3, 1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier= nn.Sequential(
            nn.Flatten(),       # combines multiple diamensional output into a single vector
            nn.Linear(
                in_features=hidden_unit*7*7,
                out_features=output_shape
            )
        )
    # forward pass
    def forward(self, x: torch.tensor):
        x= self.block_1(x)
        x= self.block_2(x)
        x= self.classifier(x)
        return x

torch.manual_seed(42)
model_2= FashionMNISTModelV2(input_shape=1,
                             hidden_unit=10,
                             output_shape= len(class_name))
# print(model_2)
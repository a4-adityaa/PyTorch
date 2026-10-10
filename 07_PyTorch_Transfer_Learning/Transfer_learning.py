" Let's start transfer learning "
import os
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torchinfo import summary

# Let's set up device agnostic code
device= "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# let's get our data
import zipfile
from pathlib import Path
import requests

# setup path to data folder
data_path = Path("Data/")
image_path= data_path / "pizza_steak_sushi"

'''f the image folder does'nt exist download it and prepare it
if image_path.is_dir():
    print(f"{image_path} directory exist..")
else:
    print(f"Did'nt find {image_path} directory.. creating one")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi
    with open(data_path/ "pizza_steak_sushi.zip", "wb") as f:
        requests = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza,steak, sushi data....")
        f.write(requests.content)

    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza_steak_sushi data....")
        zip_ref.extractall(image_path)

    #remove zip file
    os.remove(data_path / "pizza_steak_sushi.zip")'''

# now set Dirs
train_dir= image_path / "train"
test_dir= image_path / "test"

# Let's create a transform function for our data
manual_transforms= transforms.Compose([
    transforms.Resize((224,224)), # Reshape all images tp 224*224
    transforms.ToTensor(), # turns image value btwn 0 and 1
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224,0.225]) # these mean and std are the data we got from the model on which we are going to train
])

# Create training and testing Dataloader as well as get list of classes
# we will use our PyTorch going modular functions here
import sys
from pathlib import Path

# Add the folder containing Data_Setup.py
sys.path.append(
    str(Path(__file__).resolve().parent.parent / "06_PyTorch_Going_Moduler")
)

# Import the module
import Data_Setup

# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = Data_Setup.create_dataloader(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=manual_transforms, # resize, convert images to between 0 & 1 and normalize them
                                                                               batch_size=32) # set mini-batch size to 32

# print(train_dataloader, test_dataloader, class_names)

# Get a set of pretrained model weights
weights= torchvision.models.EfficientNet_B0_Weights.DEFAULT #.DEFAULT = best available weights from pretraining on ImageNet
# print(weights)

# get the transform used to create our pretrained model
auto_transform= weights.transforms()
# print(auto_transform)

# Create training and testing dataloader as well as a list of class name
train_dataloader, test_dataloader, class_names= Data_Setup.create_dataloader(train_dir=train_dir,
                                                                             test_dir=test_dir,
                                                                             transform=auto_transform,
                                                                             batch_size=32)
# print(train_dataloader, test_dataloader, class_names)
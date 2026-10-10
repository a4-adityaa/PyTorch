" Let's start transfer learning "
import os
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torchinfo import summary

# Let's set up device agnostic code
device= "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

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

" Getting a PyTorch Model... Here we are going to use efficientNet_b0 "
# OLD: Setup the model with pretrained weights and send it to the target device (this was prior to torchvision v0.13)
# model = torchvision.models.efficientnet_b0(pretrained=True).to(device) # OLD method (with pretrained=True)

# NEW: Setup the model with pretrained weights and send it to the target device (torchvision v0.13+)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# print(model) # uncomment to output (it's very long)

# Now we will print a summary of model EfficinetNet_B0 using torch.summary
'''summary(model=model,
        input_size=(32, 3, 224, 224),
        col_names=["input_size", "output_size","num_params","trainable"],
        col_width=20,
        row_settings=["var_names"])
'''

# Freeze all base layers in the "features" section of the model(the feature extractor) by setting required_grad=False
for param in model.features.parameters():
    param.requires_grad=False

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# get the kength of class name
output_shape= len(class_names)

model.classifier= torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=output_shape,
                    bias=True)
).to(device)
"Our model is updated and we will get summary again "

summary(model,
        input_size=(32, 3, 224, 224),
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

" Now train our Model "
# Define loss and optimizer
loss_fn= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr=0.001)

# set random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#start the timer
from timeit import default_timer as timer
start_time= timer()

# Setup training and save the results
import engine
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=5,
                       device=device)

end_time= timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

from helper_functions import plot_loss_curves
import matplotlib.pyplot as plt
plot_loss_curves(results)
# plt.show()

from typing import List, Tuple

from PIL import Image

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):
    
    
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False);
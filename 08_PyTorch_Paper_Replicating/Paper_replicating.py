''' Paper replicating mean using someones research and using it for learning '''

# first we will import dependencies and setup data
import torch
import torchvision
from torchinfo import summary
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from torchvision import transforms

sys.path.append(
    str(Path(__file__).resolve().parent.parent / "06_PyTorch_Going_Moduler")
)
import Data_Setup, engine

# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# setup data
data_path = Path("Data/")
image_path = data_path / "pizza_steak_sushi"

train_dir = image_path / "train"
test_dir = image_path / "test"

" Prepare transforms for images "
# create image size 
IMG_SIZE = 224
# create transform pipeline manually 
manual_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.ToTensor(),
])
# print(f"Manually created transforms: {manual_transforms}")

" Turn image into dataloader "
BATCH_SIZE = 32

# create dataloaders
train_dataloader, test_dataloader, class_name = Data_Setup.create_dataloader(
    train_dir= train_dir,
    test_dir= test_dir,
    transform=manual_transforms,
    batch_size=BATCH_SIZE
)
# print(train_dataloader, test_dataloader, class_name)


" now here we will create our custom datasets "
# dependencies
import os
import pathlib
import torch
import zipfile
from pathlib import Path
import requests

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List

# Data folder
data_path = Path("Data")
data_path.mkdir(exist_ok=True)

# Dataset folder
image_path = data_path / "pizza_steak_sushi"

# Zip file
zip_path = data_path / "pizza_steak_sushi.zip"

# Download URL
download_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"


'''# Check if dataset already exists
if not image_path.is_dir():

    print("Dataset not found. Downloading/preparing...")

    # Download only if ZIP doesn't exist
    if not zip_path.is_file():
        print("Downloading pizza, steak, sushi data...")

        request = requests.get(download_url)

        with open(zip_path, "wb") as f:
            f.write(request.content)

        print("Download complete!")

    else:
        print("Zip file already exists. Skipping download.")

    # Unzip
    print("Unzipping dataset...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(image_path)

    print("Dataset ready!")

else:
    print("Dataset already exists. Skipping download and unzip.")

print("Image path:", image_path)
print("Exists:", image_path.exists())
print("Contents:", list(image_path.rglob("*"))[:10])'''


import os
def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

train_dir= image_path / "train"
test_dir= image_path / "test"

# Setup path for target directory
target_directory = train_dir
print(f"Target directory: {target_directory}")

# Get the class names from the target directory
class_names_found = sorted([entry.name for entry in list(os.scandir(image_path / "train"))])
print(f"Class names found: {class_names_found}")


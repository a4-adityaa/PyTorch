from pathlib import Path
import requests
import zipfile
import matplotlib.pyplot as plt

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

import random
from PIL import Image

# Set seed
# random.seed(42) # <- try changing this and see what happens

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
# print(f"Random image path: {random_image_path}")
# print(f"Image class: {image_class}")
# print(f"Image height: {img.height}") 
# print(f"Image width: {img.width}")
# plt.imshow(img)
# plt.show()

" now turn our image into target tensor "

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_transform = transforms.Compose([
    # resize our image to 64 X 64
    transforms.Resize(size=(64,64)),
    # flip the image randomly in the horizonatal
    transforms.RandomHorizontalFlip(p=0.5),
    # turn data into tensor
    transforms.ToTensor()
])

def plot_transformed_images(image_path, transform, n=3, seed=42):
    random.seed(seed)

    random_image_path= random.sample(image_path, k=n)
    for image_path in random_image_path:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original Image \n size: {f.size}")
            ax[0].axiz= ("off")

            # plot transformed image
            transformed_image = transform(f).permute(1,2,0) # permute will change the color chanel to last from first
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"transformed image \n size: {transformed_image.shape}")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize= 16)

plot_transformed_images(image_path_list,
                        transform=data_transform,
                        n=3)
# plt.show()

"let's transform our data using Imagefolder"
from torchvision import datasets
train_data= datasets.ImageFolder(root=train_dir,
                                 transform=data_transform,
                                 target_transform=None)

test_data= datasets.ImageFolder(root=test_dir,
                                transform=data_transform)

# print(f"Train data: \n {train_data} \n Test data: \n{test_data}")

class_name= train_data.classes # returns classes
class_dict= train_data.class_to_idx # returns a dictionary

img,label= train_data[0][0], train_data[0][1]
# print(f"Image tensor:\n{img}")
# print(f"Image shape: {img.shape}")
# print(f"Image datatype: {img.dtype}")
# print(f"Image label: {label}")
# print(f"Label datatype: {type(label)}")
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
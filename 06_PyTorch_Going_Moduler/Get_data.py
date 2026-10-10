" Using this function We can get our data "

import os
from pathlib import Path
import requests
import zipfile

# setup path to data folder
data_path = Path("Data/")
image_path = data_path / "pizza_steak_sushi"

# if the image does'nt exist download and prepare it

if image_path.is_dir():
    print(f"{image_path} directory exist")
else:
    print(f" Didn't find {image_path} diectory, creating one....")
    image_path.mkdir(parents=True, exist_ok=True)

# Download pizza_steak_sushi data

with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request= requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print(f"Downloading pizza, steal, sushi data....")
    f.write(request.content)

# now unzip our data i.e pizza_steak_sushi
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print(f" Unzipping pizza_steak_sushi data...")
    zip_ref.extractall(image_path)

# now remove our zip file to get rid of space
os.remove(data_path / "pizza_steak_sushi.zip")
from pathlib import Path
import requests
import zipfile

# Data folder already exists / create if needed
data_path = Path("Data")
data_path.mkdir(exist_ok=True)

# Pizza, steak, sushi folder
image_path = data_path / "pizza_steak_sushi"

# Zip file
zip_path = data_path / "pizza_steak_sushi.zip"

# Download URL
download_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"


# If dataset folder doesn't exist
if not image_path.is_dir():

    print("Dataset not found. Downloading...")

    # Download only if zip doesn't already exist
    if not zip_path.is_file():
        request = requests.get(download_url)

        with open(zip_path, "wb") as f:
            f.write(request.content)

        print("Download complete!")
    else:
        print("Zip file already exists. Skipping download.")

    # Unzip
    print("Unzipping dataset...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    print("Dataset ready!")

else:
    print("Dataset already exists. Skipping download and unzip.")
from pathlib import Path
import requests
import zipfile
import matplotlib.pyplot as plt
import torch



# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

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

# rearragne the order diamension
img_permute= img.permute(1,2,0)
print(f"original shape: {img.shape} -> [color-chanel, height, width]")
print(f"Image permute shape: {img_permute.shape} -> height, width, color-chanel")

# plot
plt.figure(figsize=(10,7))
plt.imshow(img.permute(1,2,0))
plt.axis("off")
plt.title(class_name[label], fontsize=14)
# plt.show()

# now lets turn our train and test data into dataloader

from torch.utils.data import DataLoader

train_dataloader= DataLoader(dataset= train_data,
                             batch_size=1,
                             num_workers=0, # no if cpu working
                             shuffle= True)

test_dataloader= DataLoader(dataset=test_data,
                            batch_size=1,
                            num_workers=0,
                            shuffle= False)

# print(train_dataloader, test_dataloader)

# now lets make it iterable and test it

img, label= next(iter(train_dataloader))

# print(f"Image shape {img.shape} -> [batch_size, color_chanels, height, width]")
# print(f"label shape: {label.shape}")


simple_transform = transforms.Compose([ 
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
# now load and transform model
from torchvision import datasets

train_data_simple= datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple= datasets.ImageFolder(root=test_dir, transform=simple_transform )

BATCH_SIZE=32
NUM_WORKER= 0
print(f"Creating a dataloader with batchsize: {BATCH_SIZE} and {NUM_WORKER} num-worker.")

train_dataloader_simple= DataLoader(train_data_simple,
                                    batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKER,
                                    shuffle=True)

test_dataloader_simple= DataLoader(test_data_simple,
                                   batch_size= BATCH_SIZE,
                                   num_workers=NUM_WORKER,
                                   shuffle=False)

print(train_dataloader_simple, test_dataloader_simple)

" now let's create a TinyVGG model "
from torch import nn
class TinyVGG(nn.Module):

    def __init__(self, input_shape: int, hidden_unit: int, output_shape: int) -> None:
        super().__init__()

        self.conv_block_1= nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels= hidden_unit,
                    out_channels=hidden_unit,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2),
        )

        self.conv_block_2= nn.Sequential(
            nn.Conv2d(in_channels=hidden_unit,
                    out_channels=hidden_unit,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_unit,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier= nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_unit*16*16,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x= self.conv_block_1(x)
        x= self.conv_block_2(x)
        x= self.classifier(x)

        return x

torch.manual_seed(42)
model_0= TinyVGG(input_shape=3,
                 hidden_unit=10,
                 output_shape=len(train_data.classes)).to(device)

# print(model_0)

# 1. Get a batch of images and labels from the DataLoader
img_batch, label_batch = next(iter(train_dataloader_simple))

# 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
# print(f"Single image shape: {img_single.shape}\n")

# 3. Perform a forward pass on a single image
model_0.eval()
with torch.inference_mode():
    pred = model_0(img_single.to(device))
    
# 4. Print out what's happening and convert model logits -> pred probs -> pred label
# print(f"Output logits:\n{pred}\n")
# print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
# print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
# print(f"Actual label:\n{label_single}")

" Now let's create a train_step() and test_step() function "

def train_step(model= torch.nn.Module,
               dataloader= torch.utils.data.DataLoader,
               loss_fn= torch.nn.Module,
               optimizer= torch.optim.Optimizer):
    # let's train our model
    model.train()

    #let's initilize test_acc and train_acc for accumulating 
    train_acc, test_acc= 0,0

    for batch, (x,y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred= model(x)

        # setup loss_fn and accumulate it
        loss= loss_fn(y_pred,y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class= torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc= (y_pred_class==y).sum().item()/ len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

" same for test step() "

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

" now combine train_step and test_step to do training "

from tqdm.auto import tqdm

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int=5):

    results= {"train_loss: ",[],
              "train_acc: ",[],
              "test_loss: ",[],
              "test_acc:",[]}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results at the end of the epochs
    return results
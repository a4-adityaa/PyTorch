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
# plt.figure(figsize=(10,7))
# plt.imshow(img.permute(1,2,0))
# plt.axis("off")
# plt.title(class_name[label], fontsize=14)
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
    train_loss, train_acc= 0,0

    for batch, (x,y) in enumerate(dataloader):
        # Send data to target device
        x, y = x.to(device), y.to(device)

        # forward pass
        y_pred= model(x)

        # setup loss_fn and accumulate it
        loss= loss_fn(y_pred,y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class= torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/ len(y_pred)

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

    results= {"train_loss":[],
              "train_acc":[],
              "test_loss":[],
              "test_acc":[]}

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

" now train and evaluate our Model "

torch.manual_seed(42)
NUM_EPOCHS = 5

# Recreate the instacne of Model_0

model_0= TinyVGG(input_shape=3,
                 hidden_unit=10,
                 output_shape=len(train_data.classes)).to(device)

# setup loss_fn and optimizer
loss_fn= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# now train model_0

model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

end_time= timer()
print(f"Total train time: {end_time - start_time:.3f} seconds")

print(model_0_results.keys())

" Now let's plot curve to visulize "

def plot_loss_curves(results: dict[str, list[float]]):

    loss= results["train_loss"]
    test_loss= results["test_loss"]

     # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

plot_loss_curves(model_0_results)
# plt.show()

" Let's improve our model by Data Augmentation "

# create training transform
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Turn image folders into Datasets
train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)

# print(train_data_augmented, test_data_simple)

# Turn Datasets into DataLoader's
import os
BATCH_SIZE = 32
NUM_WORKERS = 0

torch.manual_seed(42)
train_dataloader_augmented = DataLoader(train_data_augmented, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple, 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=False, 
                                    num_workers=NUM_WORKERS)

print(train_dataloader_augmented, test_dataloader)

" now create another model and train it "

torch.manual_seed(42)
model_1 = TinyVGG(input_shape=3,
                 hidden_unit=10,
                 output_shape=len(train_data_augmented.classes)).to(device)

print(model_1)

" Now let's train our model_1 "

torch.manual_seed(42)
NUM_EPOCHS = 5

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Train model_1
model_1_results = train(model=model_1, 
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took

end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# now plot curves

plot_loss_curves(model_1_results)
# plt.show()

" Now compare the Results of Model_0 and Model_1 "

import pandas as pd
model_0_df= pd.DataFrame(model_0_results)
model_1_df= pd.DataFrame(model_1_results)
model_0_df

" Now plot data to visulize "

# Setup a plot 
plt.figure(figsize=(15, 10))

# Get number of epochs
epochs = range(len(model_0_df))

# Plot train loss
plt.subplot(2, 2, 1)
plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.legend()

# Plot test loss
plt.subplot(2, 2, 2)
plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.legend()

# Plot train accuracy
plt.subplot(2, 2, 3)
plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
plt.title("Train Accuracy")
plt.xlabel("Epochs")
plt.legend()

# Plot test accuracy
plt.subplot(2, 2, 4)
plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
plt.title("Test Accuracy")
plt.xlabel("Epochs")
plt.legend();

# plt.show()

" Now let's make prediction on random image "

# Download custom image
import requests

# Setup custom image path
custom_image_path = data_path / "04-pizza-dad.jpeg"

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

# now transform our image 

import torchvision

# Read in custom image
custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))

# Print out image data
print(f"Custom image tensor:\n{custom_image_uint8}\n")
print(f"Custom image shape: {custom_image_uint8.shape}\n")
print(f"Custom image dtype: {custom_image_uint8.dtype}")

# Now transform our image into the format our model trained on... else we can get error
# Load in custom image and convert the tensor values to float32
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

# Divide the image pixel values by 255 to get them between [0, 1]
custom_image = custom_image / 255. 

# Print out image data
print(f"Custom image tensor:\n{custom_image}\n")
print(f"Custom image shape: {custom_image.shape}\n")
print(f"Custom image dtype: {custom_image.dtype}")

# now plot custom image
# Plot custom image
plt.imshow(custom_image.permute(1, 2, 0)) # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
plt.title(f"Image shape: {custom_image.shape}")
plt.axis(False);
# plt.show()

# Create transform pipleine to resize image
custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
])

# Transform target image
custom_image_transformed = custom_image_transform(custom_image)

# Print out original shape and new shape
print(f"Original shape: {custom_image.shape}")
print(f"New shape: {custom_image_transformed.shape}")

# now add batch size else we can get error
model_1.eval()
with torch.inference_mode():
    # Add an extra dimension to image
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
    
    # Print out different shapes
    print(f"Custom image transformed shape: {custom_image_transformed.shape}")
    print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")
    
    # Make a prediction on image with an extra dimension
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))

print(custom_image_pred)

# Print out prediction logits
print(f"Prediction logits: {custom_image_pred}")

# Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
print(f"Prediction probabilities: {custom_image_pred_probs}")

# Convert prediction probabilities -> prediction labels
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
print(f"Prediction label: {custom_image_pred_label}")

# Find the predicted label
custom_image_pred_class = class_name[custom_image_pred_label.cpu()] # put pred label to CPU, otherwise will error
custom_image_pred_class


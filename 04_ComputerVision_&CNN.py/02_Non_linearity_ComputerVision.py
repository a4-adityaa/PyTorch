" Let's create a non-linear model "
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

" Now create some data "

train_data = datasets.FashionMNIST(
    root="Data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data= datasets.FashionMNIST(
    root="Data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

image,label= train_data[0]
# print(image, label)
# print(len(train_data), len(train_data.targets), len(test_data), len(test_data.targets))

"prepare data loader "

from torch.utils.data import DataLoader
BATCH_SIZE=32

train_dataloader= DataLoader(train_data,
                             batch_size= BATCH_SIZE,
                             shuffle= True)

test_dataloader= DataLoader(test_data,
                            batch_size= BATCH_SIZE,
                            shuffle=False)

# # Let's check out what we've created
# print(f"Dataloaders: {train_dataloader, test_dataloader}") 
# print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
# print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

class_name = train_data.classes
class_idx= train_data.class_to_idx

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

" Now create a model "

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_unit: int, output_shape: int):
        super().__init__()
        self.layer_stack= nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_unit),
            nn.ReLU(), # -> non linear activation function
            nn.Linear(in_features=hidden_unit, out_features=hidden_unit),
            nn.ReLU(),
            nn.Linear(in_features=hidden_unit, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

" Now create instace of model "
model_1= FashionMNISTModelV1(input_shape= 784,
                             hidden_unit=128,
                             output_shape=len(class_name))

print(model_1)

" setup loss, optimizer and accuracy "
from helper_functions import accuracy_fn
loss_fn= nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(params=model_1.parameters(), lr=0.1)

''' Ok now we are going to functionize our train and test loop to get rid of writing code agarin and againn 
training loop as train step 
testing loop as test loop  '''

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):

    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

torch.manual_seed(42)

# Measure time
from timeit import default_timer as timer
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    # print(f"Train time on {device}: {total_time:.3f} seconds") # .3f mean the output upto 3 decimal places
    return total_time
train_time_start_on_gpu = timer()
from tqdm import tqdm

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_1, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )
    test_step(data_loader=test_dataloader,
        model=model_1,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)

torch.manual_seed(42)
from helper_functions import eval_model
# Note: This will error due to `eval_model()` not using device agnostic code 
model_1_results = eval_model(model=model_1, 
    data_loader=test_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn) 
print(model_1_results) 
import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

" We are going to use FashionMNIST for geeting data "

train_data= datasets.FashionMNIST(
    root= "Data", # -> where we want data
    train= True, # want traing data or not (gives training data)
    download= True, # want to download
    transform= torchvision.transforms.ToTensor(), # hwo do we want to tranform data
    target_transform= None # how we want to transform lables/targets
)

test_data= datasets.FashionMNIST(
    root="Data",
    train=False,
    download= True,
    transform= ToTensor(), # can be also written like this
    target_transform= None
)
 
# print(len(train_data), len(test_data))

# see the first training data
image, label= train_data[0]
# print(image,label)
# print(image.shape) # -> shape of our data
# print(len(train_data),len(train_data.targets), len(test_data),len(test_data.targets)) # -> Numbers of samples

 # see classes
class_name = train_data.classes
# print(class_name) # -> gives the total classes

class_idx= train_data.class_to_idx
# print(class_idx) # -> gives dictionary of classes and index

# Now Visulize
# image, label = train_data[0]
# print(f"Image shape: {image.shape}")
# plt.imshow(image.squeeze(), cmap="grey") # image shape is [1, 28, 28] (colour channels, height, width)
# plt.title(class_name[label]);
# plt.axis(False)
# plt.show()

# Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
'''
for i in range(1, rows * cols + 1): # loops start from 1 and goes to (4*4+1) i.e 17 can say 16 images 
    random_idx = torch.randint(0, len(train_data), size=[1]).item() # selects the random data from the dataset and .item is used to show data in integer form
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i) # this plots the data on every index of i
    plt.imshow(img.squeeze(), cmap="gray") # removes extra diamensions and cmap="gery" -> refers to color 
    plt.title(class_name[label]) # show titles
    plt.axis(False); # hides tha axis
plt.show()
'''

from torch.utils.data import DataLoader
BATCH_SIZE=32

train_dataloader= DataLoader(train_data, # the data we want to iterate
                             batch_size=BATCH_SIZE, # no of elemnts at once we want to give to Model
                             shuffle= True) # shuffle data ever epochs

test_dataloader= DataLoader(test_data,
                            batch_size=BATCH_SIZE,
                            shuffle=False) # do not need to necceserily shuffle data

# let's check the number of dataloader
# print(f"the no of train dataloader: {len(train_dataloader)} in batches of: {BATCH_SIZE}")
# print(f"the no of test dataloader: {len(test_dataloader)} in batches of: {BATCH_SIZE}")

# Checks whats inside training dataloader
train_features_batch,train_label_batch =next(iter(train_dataloader))
# print(train_features_batch.shape, train_label_batch.shape)

# now visulize the train_features_batch dataloader

for i in range(1, rows*cols+1):
    random_idx= torch.randint(0, len(train_label_batch), size=[1]).item()
    img, label= train_features_batch[random_idx], train_label_batch[random_idx]
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_name[label])
    plt.axis(False)
# plt.show()

'''  Now let's create a model '''

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_unit: int, output_shape: int):
        super().__init__()
        self.layer_stack= nn.Sequential(
            nn.Flatten(), # -> flatten the input data
            nn.Linear(in_features=input_shape, out_features=hidden_unit), # -> first linear layer
            nn.Linear(in_features=hidden_unit, out_features=hidden_unit), # -> second linear layer
            nn.Linear(in_features=hidden_unit, out_features=output_shape) # -> third linear layer
        )

    # forward pass
    def forward(self, x):
        return self.layer_stack(x)

# let's define models instace
model_0= FashionMNISTModelV0(input_shape=784,
                             hidden_unit=10,
                             output_shape=len(class_name))

from helper_functions import accuracy_fn
loss_fn= nn.CrossEntropyLoss()
optimzer= torch.optim.SGD(params=model_0.parameters(),lr=0.1)

# create a time function to calculate the total time to run our model

from timeit import default_timer as timer
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    # print(f"Train time on {device}: {total_time:.3f} seconds") # .3f mean the output upto 3 decimal places
    return total_time

''' Now let's create training and testing loop '''

torch.manual_seed(42)
train_time_start_on_cpu = timer()

from tqdm.auto import tqdm
epochs = 3

for epoch in tqdm(range(epochs)): # tqdm is used to show the progress bar i.e how much training is completed
    # print(f"epoch: {epochs}\n..........")

    train_loss = 0 # will keep tracl of train loss per batches

    for batch, (X,y) in enumerate(train_dataloader): # enumurate takes class name with their data
        ### Training
        model_0.train() # starts the training
        y_pred= model_0(X) # forward passs : y_preds mean predicted data by the model

        loss= loss_fn(y_pred, y) # compares predicted data with original label i.e y
        train_loss += loss # add loss per batch

        optimzer.zero_grad() # clears the tracking
        loss.backward() # BackPropagation
        optimzer.step() # optimizer step

        # if batch % 400 ==0:
        #     print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    #Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)

    ### Testing
    test_loss, test_acc= 0,0
    model_0.eval()
    with torch.inference_mode():
        for X,y in test_dataloader: # takes the data i.e train and test 
            test_pred = model_0(X) # forward pass the train data

            test_loss += loss_fn(test_pred, y) # stores the test loss per epoch
            test_acc += accuracy_fn(y_true= y, y_pred= test_pred.argmax(dim=1)) # stores the accuracy per epoch

        test_loss /= len(test_dataloader) # calculate avg test loss
        test_acc /= len(test_dataloader) # calculate avg test accuracy

    # print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

# let's calculate the time;
train_time_end_on_cpu = timer()
total_train_time_model_0= print_train_time(start=train_time_start_on_cpu,
                                           end=train_time_end_on_cpu,
                                           device=str(next(model_0.parameters())))

" Now let's create a eval function so that we don't have to write testing code all the time "

torch.manual_seed(42)

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.dataloader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    loss,acc=0,0
    model.eval()

    with torch.inference_mode():
        for X,y in  data_loader:
            y_pred= model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# Calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn
)
print(model_0_results)
" Trains a PyTorch image classification model using device agnostic code "

import os
import torch
import Data_Setup, engine, Model_builder, utils

from torchvision import transforms

# set hyperparameters
NUM_EPOCHS= 5
BATCH_SIZE= 32
HIDDEN_UNIT= 10
LEARNING_RATE= 0.001

# Set directories
train_dir= "Data/pizza_steak_sushi/train"
test_dir= "Data/pizza_steak_sushi/test"

# Setup decice agnpstic code
device= "cuda" if torch.cuda.is_available() else "cpu"

# create transform
data_trnsform= transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# create transforms
train_dataloader, test_dataloader, class_name= Data_Setup.create_dataloader(train_dir=train_dir,
                                                                            test_dir=test_dir,
                                                                            transform=data_trnsform,
                                                                            batch_size=BATCH_SIZE)

# Create model with help from model_builder.py
model= Model_builder.TinyVGG(input_shape=3,
                             hidden_unit=HIDDEN_UNIT,
                             output_shape=len(class_name)) 

# set loss and optimizer
loss_fn= torch.nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(params=model.parameters(),
                            lr=LEARNING_RATE)

# start training with help of engine engine.py

engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=NUM_EPOCHS,
             device=device)

# save model using utils.py
utils.save_model(model=model,
                 target_dir="Models",
                 model_name="05_going_modular_tinyVgg_model.pth")
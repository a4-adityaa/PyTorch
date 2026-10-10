'''Contains functions for training and testing PyTorch Model '''

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    ''' Trains a PyTorch Model for a single epoch

    Turns a target PyTorch model to training mode and then runs
    through all the required traing steps (forward pass, loss calculation, optimzer step).

    Args: 
        model: A PyTorch model to be trained.
        dataloader: A dataloader instance for the model to be trained on.
        loss_fn: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu") 

    Returns:
        A tuple of training loss and training accuarcy metrics,
        in the form of (train_loss, train_accuarcy),

        for ex: (0.1121, 0.7863)
    '''

    # Put model into traiing mode
    model.train()

    # setup train loss and train accuracy values
    train_loss, train_acc= 0,0

    # loop through dataloader, data batches
    for batch,(X,y) in enumerate(dataloader):
        # send data to target device
        X,y= X.to(device), y.to(device)

        # forward pass
        y_pred= model(X)

        # calculate and accumulate loss
        loss= loss_fn(y_pred, y)
        train_loss += loss.item()

        # Optimize zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # calculate and accumulate accuarcy metrics accross all batches
        y_pred_class= torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/ len(y_pred)

    # Adjust Metrics to get average loss and accurcay per batch
    train_loss= train_loss/ len(dataloader)
    train_acc= train_acc/ len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    ''' Tests a PyTorch Model for a single epoch 
    
    Turns a PyTorch into "eval" mode and then perform the forward pass on a testing dataset

    Args:
        model: A PyTorch model.
        dataloader: A dataloader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute

    Returns:
        A tuple of testing loss and testing accuacry metrics
        in the form of (test_loss, test_accuracy). for ex

        (0.2130, 0.5341)
    '''
    # put model on evaluation mode
    model.eval()

    # setup test loss and test accuracy values
    test_loss, test_acc= 0,0

    # turn on infrence context manager
    with torch.inference_mode():
        # loop throgh dataloader batches
        for batch, (x,y) in enumerate(dataloader):
            # sends data to target device
            X,y= x.to(device), y.to(device)

            # forward pass
            test_pred_logits= model(X)

            # calculate and accumuate loss
            loss= loss_fn(test_pred_logits,y)
            test_loss += loss.item()

            # calculate and accumulate accuracy
            test_pred_lables= test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_lables == y).sum().item() / len(y))

    # Adjust metrics to get avegare test loss and test accuracy
    test_loss= test_loss/ len(dataloader)
    test_acc= test_acc / len(dataloader)

    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str,str]:
    '''Trains and test a PyTorch Model 

    Passes a target PyTorch models through train_step() and test_step() functions
    for a number of epochs, training and testing the model in the same epoch loop.

    Calculate, prints and stores evaluations metrics throughout.

    Args:
        model: A model to be tarined and tested.
        datalaoder: A dataloader instace for the model to be tested on.
        optimizer: A PyTorch Optimizer to help minimize loss.
        loss_fn: A PyTorch loss function to calcilate loss on both datasets,
        epochs: An integer indiacting how much epochs to be trained for.
        device: A target device to compute 

    Returns:
        A dictionary of training and testing loss as well as training and testing accuarcy metrucs.
        Each metrics has a value in list for each epochs.

        in the form of: {train_loss:[...],
                         train_acc:[...],
                         test_loss:[...],
                         test_acc:[...]}

        For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
''' 
    # Create empty dictionay list
    results= {"train_loss":[],
              "train_acc":[],
              "test_loss":[],
              "test_acc":[]}

    # loop through training and testing steps for a number of epochs:
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc= train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)

        test_loss, test_acc= test_step(model=model,
                                       dataloader=test_dataloader,
                                       loss_fn=loss_fn,
                                       device=device)

        # prints what happening
        print(
            f"Epoch: {epoch+1} |"
            f"train_loss: {train_loss:.4f} |"
            f"train_acc: {train_acc:.4f} |"
            f"test_acc:{test_acc:.4f} |"
        )

        #update result dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Returns result dictonary
    return results
'''Here we will setup our data ........
    Contains functionality like Creating Pytorch DataLoader for image classification
'''
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS= 0  #we can write os.cpu_count() but due to acchitecture issue i can't

def create_dataloader(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_worker: int=NUM_WORKERS
):
    '''Creates a training directory and testing Dataloaders

     Takes in training directory and testing directory path and turns them into
    Pytorch Datasets and then intoo PyTorch Dataloaders

    Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: Torchvision transform to perform on training data and testing data.
    batch_size: Number of samples per batch in each Dataloader.
    num_worker: An integer for number of worker(cpu or gpu) per Dataloader

    Returns:
    A tuple of (train_dataloader, test_dataloader, class_name).
    where class_name is a list of target classes.

    Example Usage:
    train_dataloader, test_dataloader, class_name = \
    = create_dataloader(train_dir= path/to/train_dir,
                        test_dir= path/to/test_dir,
                        transform= some_transform,
                        batch_size= 32,
                        num_worker= 4)
    '''

    # Get class name
    class_names= train_data.classes

    # Use image folder to create data
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # turn image into dataloader
    train_dataloader= DataLoader(train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_worker,
                                 pin_memory=True) # data transfer optimization btwn cpu to gpu

    test_dataloader= DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=False, # No need to shuffle test data
                                num_workers=num_worker,
                                pin_memory=True)

    return train_dataloader, test_dataloader, class_names

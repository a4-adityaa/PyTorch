" Contains various utility functions for PyTorch model traininga and saving "

import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    '''saves a pyTorch Model to a target diectory
    Args:
        model: A target PyTorch model to save.
        target_dir: A directory to save model.
        model_name: A file name for the saved model.. should include
        either ".pth" or ".pt" as the file extension.

    Example Usage:
        save_model(model=model_0,
                    target_dir="models",
                    model_name="model_0.pth")
    '''

    # create target directory
    target_dir_path= Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # create and save model
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name shuold end with '.pt' or '.pth'"
    model_save_path= target_dir_path / model_name

    # Saving the model state dict
    print(f"[INFO] Saving model to : {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)

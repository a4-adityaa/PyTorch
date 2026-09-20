import torch
from pathlib import Path

from training_saving_model import linearregressionModel

loaded_model_0 = linearregressionModel()

loaded_model_0.load_state_dict(
    torch.load(f=Path("Models/Model_0.pth"))
)

print(loaded_model_0.state_dict())
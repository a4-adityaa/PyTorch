''' Contains PyTorch model code to instantiate a TonyVGG Model '''

import torch
from torch import nn

class TinyVGG(nn.Module):
    '''Creates a TinyVGG architecture
    
    Replicates the TinyVGG architecture from the CNN explianer website in PyTorch

    Args:
    input_size: An integer indicting number of input chanels.
    hidden_unit: An integer indicating number of hidden unit between layers,
    output_shape: An integer indicating number of output units.
    '''

    def __init__(self, input_shape: int, hidden_unit: int, output_shape: int) -> None:
        super.__init__()
        self.conv_block_1= nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      stride=1,
                      padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_unit,
                  out_channels=hidden_unit,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
        )

        self.conv_block_2= nn.Sequential(
            nn.Conv2d(in_channels=hidden_unit,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_unit,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_unit,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x= self.conv_block_1(x)
        x= self.conv_block_2
        x= self.classifier(x)
        return x
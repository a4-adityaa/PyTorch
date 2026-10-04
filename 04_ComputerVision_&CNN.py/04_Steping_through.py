''' Here we are going through nn.Conv2d() and nn.MaxPool2d() '''
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision

torch.manual_seed(42)

images= torch.randn(size=(32,3,64,64)) # batch_size, color_chanel, height, width
test_image= images[0]
# print(f"Image batch shape: {images.shape} -> [batch_size, color_channels, height, width]")
# print(f"Single image shape: {test_image.shape} -> [color_channels, height, width]") 
# print(f"Single image pixel values:\n{test_image}")

torch.manual_seed(42)

" First we will try Conv2d() layer "
# Create a convolutional layer with same dimensions as TinyVGG 
# (try changing any of the parameters and see what happens)
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0) # also try using "valid" or "same" here 

# Pass the data through the convolutional layer
# print(conv_layer(test_image))

# Add extra dimension to test image
test_image.unsqueeze(dim=0).shape

# Pass test image with extra dimension through conv_layer
conv_layer(test_image.unsqueeze(dim=0)).shape # add print to show values

torch.manual_seed(42)
# Create a new conv_layer with different values (try setting these to whatever you like)
conv_layer_2 = nn.Conv2d(in_channels=3, # same number of color channels as our input image
                         out_channels=10,
                         kernel_size=(5, 5), # kernel is usually a square so a tuple also works
                         stride=2,
                         padding=0)

# Pass single image through new conv_layer_2 (this calls nn.Conv2d()'s forward() method on the input)
conv_layer_2(test_image.unsqueeze(dim=0)).shape

" Now let's test MaxPool2d() layer "
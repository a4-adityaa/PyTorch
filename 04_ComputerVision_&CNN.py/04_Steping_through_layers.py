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

# Print out original image shape without and with unsqueezed dimension
# print(f"Test image original shape: {test_image.shape}")
# print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}")

# Create a sample nn.MaxPoo2d() layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass data through just the conv_layer
test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
# print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")

# Pass data through the max pool layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
# print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")

" now we will create a small tesnor and try on it "
torch.manual_seed(42)
# Create a random tensor with a similar number of dimensions to our images
random_tensor = torch.randn(size=(1, 1, 2, 2))
# print(f"Random tensor:\n{random_tensor}")
# print(f"Random tensor shape: {random_tensor.shape}")

# Create a max pool layer
max_pool_layer = nn.MaxPool2d(kernel_size=2) # see what happens when you change the kernel_size value 

# Pass the random tensor through the max pool layer
max_pool_tensor = max_pool_layer(random_tensor)
# print(f"\nMax pool tensor:\n{max_pool_tensor} <- this is the maximum value from random_tensor")
# print(f"Max pool tensor shape: {max_pool_tensor.shape}")
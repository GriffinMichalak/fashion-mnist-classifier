import cv2
import numpy
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *


conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))

# Get the weights of the first convolutional layer of the network
conv1_weights = conv_net.conv1.weight
print("Weights of the first convolutional layer:")
print(conv1_weights)


# Create a plot that is a grid of images, where each image is one kernel from the conv layer.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, 
# the grid might have 4 rows and 8 columns. Finally, normalize the values in the grid to be 
# between 0 and 1 before plotting.

''' YOUR CODE HERE '''



# Save the grid to a file named 'kernel_grid.png'. Add the saved image to the PDF report you submit.

''' YOUR CODE HERE '''




# Apply the kernel to the provided sample image.

img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0					# Normalize the image
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)

print(img.shape)

# Apply the kernel to the image

output = ''' YOUR CODE HERE '''


# convert output from shape (1, num_channels, output_dim_0, output_dim_1) to (num_channels, 1, output_dim_0, output_dim_1) for plotting.
# If not needed for your implementation, you can remove these lines.

output = output.squeeze(0)
output = output.unsqueeze(1)


# Create a plot that is a grid of images, where each image is the result of applying one kernel to the sample image.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, the grid might have 4 rows and 8 columns.
# Finally, normalize the values in the grid to be between 0 and 1 before plotting.

''' YOUR CODE HERE '''



# Save the grid to a file named 'image_transform_grid.png'. Add the saved image to the PDF report you submit.

''' YOUR CODE HERE '''


# Create a feature map progression. You can manually specify the forward pass order or programatically track each activation through the forward pass of the CNN.

''' YOUR CODE HERE '''

# Save the image as a file named 'feature_progression.png'

''' YOUR CODE HERE '''













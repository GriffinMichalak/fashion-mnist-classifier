import cv2
import numpy
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
import matplotlib.pyplot as plt
import numpy as np


conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))

# Get the weights of the first convolutional layer of the network
conv1_weights = conv_net.conv1.weight
print("Weights of the first convolutional layer:")


# Create a plot that is a grid of images, where each image is one kernel from the conv layer.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, 
# the grid might have 4 rows and 8 columns. Finally, normalize the values in the grid to be 
# between 0 and 1 before plotting.

rows = 4
cols = 8
plt.figure(figsize=(5,5))
for i in range(len(conv1_weights)):
    plt.subplot(rows, cols, i+1)
    kernel = conv1_weights[i]
    image_np = kernel.detach().numpy().squeeze(0)
    min_val = np.min(image_np)
    max_val = np.max(image_np)
    normalized = (image_np - min_val) / (max_val - min_val)
    plt.imshow(normalized, cmap=plt.cm.binary)
    plt.axis('off')
plt.show()

# Save the grid to a file named 'kernel_grid.png'. Add the saved image to the PDF report you submit.
plt.savefig("kernel_grid.png") 


# Apply the kernel to the provided sample image.
img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0					# Normalize the image
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)

print(img.shape)

# Apply the kernel to the image

output = F.conv2d(img, conv1_weights, bias=None, stride=1, padding=0)


# convert output from shape (1, num_channels, output_dim_0, output_dim_1) to (num_channels, 1, output_dim_0, output_dim_1) for plotting.
# If not needed for your implementation, you can remove these lines.

output = output.squeeze(0)
output = output.unsqueeze(1)


# Create a plot that is a grid of images, where each image is the result of applying one kernel to the sample image.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, the grid might have 4 rows and 8 columns.
# Finally, normalize the values in the grid to be between 0 and 1 before plotting.

plt.figure(figsize=(5,5))
for i in range(len(output)):
    plt.subplot(rows, cols, i+1)
    kernel = output[i]
    image_np = kernel.detach().numpy().squeeze(0)
    min_val = np.min(image_np)
    max_val = np.max(image_np)
    normalized = (image_np - min_val) / (max_val - min_val)
    plt.imshow(normalized, cmap=plt.cm.binary)
    plt.axis('off')
plt.show()

# Save the grid to a file named 'image_transform_grid.png'. Add the saved image to the PDF report you submit.
plt.savefig("image_transform_grid.png") 


# Create a feature map progression. You can manually specify the forward pass order or programatically track each activation through the forward pass of the CNN.
feature_maps = []

# Layer 1: conv1 -> ReLU -> Pool
x = conv_net.conv1(img)
x = conv_net.activ_fn(x)
feature_maps.append(x.squeeze(0))
x = conv_net.pool(x)

# Layer 2: conv2 -> ReLU -> Pool
x = conv_net.conv2(x)
x = conv_net.activ_fn(x)
feature_maps.append(x.squeeze(0))
x = conv_net.pool(x)

# Layer 3: conv3 -> ReLU -> Pool
x = conv_net.conv3(x)
x = conv_net.activ_fn(x)
feature_maps.append(x.squeeze(0))

rows = 1
cols = 3
plt.figure(figsize=(15, 5))
for layer_idx in range(len(feature_maps)):
    plt.subplot(rows, cols, layer_idx + 1)
    layer_output = feature_maps[layer_idx]
    image_np = layer_output.mean(dim=0).detach().numpy()
    min_val = np.min(image_np)
    max_val = np.max(image_np)
    if max_val > min_val:
        normalized = (image_np - min_val) / (max_val - min_val)
    else:
        normalized = image_np
    
    plt.imshow(normalized, cmap=plt.cm.binary)
    plt.title(f'Layer {layer_idx + 1} ({layer_output.shape[0]} channels)')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Save the image as a file named 'feature_progression.png'
plt.savefig("feature_progression.png") 
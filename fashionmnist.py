import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
from ffn import *

'''

In this file you will write end-to-end code to train two neural networks to categorize fashion-mnist data,
one with a feedforward architecture and the other with a convolutional architecture. You will also write code to
evaluate the models and generate plots.

'''


'''

PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted. Please do not change the transforms given below - the autograder assumes these.

'''

transform = transforms.Compose([                            # Use transforms to convert images to tensors and normalize them
    transforms.ToTensor(),                                  # convert images to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])             # Common method for grayscale images
])

batch_size = ''' Insert a good batch size number here '''


'''

PART 2:
Load the dataset. Make sure to utilize the transform and batch_size from the last section.

'''

trainset = torchvision.datasets.FashionMNIST(''' Fill in this function ''')
trainloader = torch.utils.data.DataLoader(''' Fill in this function ''')

testset = torchvision.datasets.FashionMNIST(''' Fill in this function ''')
testloader = torch.utils.data.DataLoader(''' Fill in this function ''')


'''

PART 3:
Complete the model defintion classes in ffn.py and cnn.py. We instantiate the models below.

'''


feedforward_net = FF_Net()
conv_net = Conv_Net()



'''

PART 4:
Choose a good loss function and optimizer - you can use the same loss for both networks.

'''

criterion = ''' Choose a good loss function '''

optimizer_ffn = ''' Choose a good optimizer and its hyperparamters for the feedforward network'''
optimizer_cnn = ''' Choose a good optimizer and its hyperparamters for the convolutional network'''



'''

PART 5:
Train both your models, one at a time! (You can train them simultaneously if you have a powerful enough computer,
and are using the same number of epochs, but it is not recommended for this assignment.)

'''


num_epochs_ffn = '''Choose the number of epochs for the feedforward network'''

for epoch in range(num_epochs_ffn):  # loop over the dataset multiple times
    running_loss_ffn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Flatten inputs for ffn

        ''' YOUR CODE HERE '''

        # zero the parameter gradients
        optimizer_ffn.zero_grad()

        # forward + backward + optimize
        outputs = feedforward_net(inputs)
        loss = criterion(outputs, '''YOUR CODE HERE''')
        loss.backward()
        optimizer_ffn.step()
        running_loss_ffn += loss.item()

    print(f"Training loss: {running_loss_ffn}")

print('Finished Training')

torch.save(feedforward_net.state_dict(), 'ffn.pth')  # Saves model file (upload with submission)


num_epochs_cnn = '''Choose the number of epochs for the convolutional network'''

for epoch in range(num_epochs_cnn):  # loop over the dataset multiple times
    running_loss_cnn = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer_cnn.zero_grad()

        # forward + backward + optimize
        outputs = conv_net(inputs)
        loss = criterion(outputs, '''YOUR CODE HERE''')
        loss.backward()
        optimizer_cnn.step()
        running_loss_cnn += loss.item()

    print(f"Training loss: {running_loss_cnn}")

print('Finished Training')

torch.save(conv_net.state_dict(), 'cnn.pth')  # Saves model file (upload with submission)


'''

PART 6:
Evalute your models! Accuracy should be greater or equal to 80% for both models.

Code to load saved weights commented out below - may be useful for debugging.

'''

# feedforward_net.load_state_dict(torch.load('ffn.pth'))
# conv_net.load_state_dict(torch.load('cnn.pth'))

correct_ffn = 0
total_ffn = 0

correct_cnn = 0
total_cnn = 0

with torch.no_grad():           # since we're not training, we don't need to calculate the gradients for our outputs
    for data in testloader:

        '''
        
            YOUR CODE HERE

        '''

print('Accuracy for feedforward network: ', correct_ffn/total_ffn)
print('Accuracy for convolutional network: ', correct_cnn/total_cnn)


'''

PART 7:

Check the instructions PDF. You need to generate some plots. 

'''

'''

YOUR CODE HERE

'''

'''
PART 8:
Compare the performance and characteristics of FFN and CNN models.
'''

'''

YOUR CODE HERE

'''
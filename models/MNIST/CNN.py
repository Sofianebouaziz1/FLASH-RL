import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_channels, out_channels1, out_channels2, num_classes):
        
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = out_channels1, kernel_size=5, stride = 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv2 = nn.Conv2d(in_channels = out_channels1, out_channels = out_channels2, kernel_size=5, stride = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self,x):
        #first Convolutional layers
        x = self.conv1(x)
        #activation function 
        x = F.relu(x)
        #max pooling 
        x = self.maxpool1(x)
        #first Convolutional layers
        x = self.conv2(x)
        #activation function
        x = F.relu(x)
        #max pooling
        x = self.maxpool2(x)
        #flatten output 
        x = torch.flatten(x,1)
        #fully connected layer 1
        x =self.fc1(x)
        #activation function
        x = F.relu(x)
        #fully connected layer 2
        x = self.fc2(x)
        # get log probabilities
        #x = F.log_softmax(x, dim=1)
        return x
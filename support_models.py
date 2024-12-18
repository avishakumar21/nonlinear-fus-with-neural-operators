import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  

        self.fc1 = nn.Linear(128 * 24 * 76, 1024)  
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  
        x = self.pool(self.relu(self.conv2(x))) 
        x = self.pool(self.relu(self.conv3(x)))  
        
        x = x.view(x.size(0), -1)  # Flatten to [bz, 128 * 24 * 76]
        
        x = self.relu(self.fc1(x))  # [bz, 1024]
        x = self.fc2(x)  # [bz, 512]
        
        return x


class FullyConnectedBranch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedBranch, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

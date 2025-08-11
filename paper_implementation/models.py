import torch.nn as nn
import torch.nn.functional as F
from config import CLASSIFIER_INPUT_SIZE

class TumourClassifier(nn.Module):
    """
    A simple Convolutional Neural Network for image classification,
    based on the paper's description.
    """
    def __init__(self, num_classes):
        super(TumourClassifier, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 20 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer: 20 input channels, 10 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the flattened size after pooling layers
        flattened_size = 10 * (CLASSIFIER_INPUT_SIZE // 4) * (CLASSIFIER_INPUT_SIZE // 4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
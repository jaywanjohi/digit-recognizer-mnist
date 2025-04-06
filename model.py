import torch.nn as nn
import torch.nn.functional as F

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 28x28 -> 26x26
        x = F.max_pool2d(x, 2) # 26x26 -> 13x13
        x = F.relu(self.conv2(x)) # 13x13 -> 11x11
        x = F.max_pool2d(x, 2) # 11x11 -> 5x5
        x = x.view(-1, 32*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

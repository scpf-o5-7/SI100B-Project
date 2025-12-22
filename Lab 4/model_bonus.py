import torch
import torch.nn as nn

class SI100FaceNet_Bonus(nn.Module):
    def __init__(self, printtoggle=False):
        super().__init__()
        self.print = printtoggle

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu3 = nn.LeakyReLU(negative_slope=0.01)

        self.fc = nn.Linear(256 * 6 * 6, 7)
        self.relu_fc = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):

        x = self.conv1(x)
        if self.print: print(f"After conv1: {x.shape}")
        x = self.pool1(x)
        if self.print: print(f"After pool1: {x.shape}")
        x = self.relu1(x)

        x = self.conv2(x)
        if self.print: print(f"After conv2: {x.shape}")
        x = self.pool2(x)
        if self.print: print(f"After pool2: {x.shape}")
        x = self.relu2(x)

        x = self.conv3(x)
        if self.print: print(f"After conv3: {x.shape}")
        x = self.pool3(x)
        if self.print: print(f"After pool3: {x.shape}")
        x = self.relu3(x)
        
        x = torch.flatten(x, start_dim=1)
        if self.print: print(f"After flatten: {x.shape}")

        x = self.fc(x)
        if self.print: print(f"After fc: {x.shape}")
        x = self.relu_fc(x)
        
        return x
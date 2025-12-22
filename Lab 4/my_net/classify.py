import torch
import torch.nn as nn
import torch.nn.functional as F

class emotionNet(nn.Module):
    def __init__(self, printtoggle):
        super().__init__()
        self.print = printtoggle
      
        # step1: Define the functions you need
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Activation function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 4 * 4, out_features=512)  # 根据图2计算的特征图尺寸
        self.fc2 = nn.Linear(in_features=512, out_features=3)  # 3个输出类别：happy, sad, neutral

    def forward(self, x):
        if self.print: print(f"Input shape: {x.shape}")
        
        # First block: convolution -> maxpool -> leaky_relu
        x = self.conv1(x)
        if self.print: print(f"After conv1: {x.shape}")
        x = self.pool(x)
        if self.print: print(f"After pool1: {x.shape}")
        x = self.leaky_relu(x)
        
        # Second block: convolution -> maxpool -> leaky_relu
        x = self.conv2(x)
        if self.print: print(f"After conv2: {x.shape}")
        x = self.pool(x)
        if self.print: print(f"After pool2: {x.shape}")
        x = self.leaky_relu(x)
        
        # Third block: convolution -> maxpool -> leaky_relu
        x = self.conv3(x)
        if self.print: print(f"After conv3: {x.shape}")
        x = self.pool(x)
        if self.print: print(f"After pool3: {x.shape}")
        x = self.leaky_relu(x)
        
        # Flatten for linear layers
        x = torch.flatten(x, start_dim=1)
        if self.print: print(f"After flatten: {x.shape}")
        
        # Fully connected layers with activation
        x = self.fc1(x)
        x = self.leaky_relu(x)
        if self.print: print(f"After fc1: {x.shape}")
        
        x = self.fc2(x)
        if self.print: print(f"Output shape: {x.shape}")
        
        return x

def makeEmotionNet(printtoggle=False):
    model = emotionNet(printtoggle)

    # L_{CE} loss function
    lossfun = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)

    return model, lossfun, optimizer
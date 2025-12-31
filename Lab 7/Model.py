import torchvision.models as models
import torch.nn as nn


class SI100FaceNet(nn.Module):
    def __init__(self, num_classes=3, printtoggle=False):
        super().__init__()
        self.print = printtoggle
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        if self.print:
            print(f"Output shape: {x.shape}")
        return x

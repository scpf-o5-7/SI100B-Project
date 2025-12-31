import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet18_Weights


class SI100BFaceNet(nn.Module):
    def __init__(self, num_classes=3, freeze_strategy="progressive", printtoggle=False):
        super().__init__()
        self.print = printtoggle
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        if freeze_strategy == "progressive":
            for name, param in self.resnet.named_parameters():
                if "layer3" in name or "layer4" in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

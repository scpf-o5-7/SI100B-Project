import torchvision.models as models
import torch.nn as nn


class SI100BFaceNet(nn.Module):
    def __init__(self, num_classes=3, freeze_strategy="progressive"):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        if freeze_strategy == "progressive":
            for name, param in self.resnet.named_parameters():
                if 'layer3' in name or 'layer4' in name or 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        if self.print:
            print(f"Output shape: {x.shape}")
        return x

#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: Zhenghao Li
# Date: 2024-11-08

import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F

class emotionNet(nn.Module):
    def __init__(self, printtoggle):
        super().__init__()
        self.print = printtoggle
      
        ### write your codes here ###
        #############################
        # step1:
        # Define the functions you need: convolution, pooling, activation, and fully connected functions.


    def forward(self, x):
        #Step 2
        # Using the functions your defined for forward propagate
        # First block
        # convolution -> maxpool -> relu

        # Second block
        # convolution -> maxpool -> relu

        # Third block
        # convolution -> maxpool -> relu

        # Flatten for linear layers

        # fully connect layer

        return x

def makeEmotionNet(printtoggle=False):
    model = emotionNet(False)

    #L_{CE} loss function
    lossfun = nn.CrossEntropyLoss()
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = .001, weight_decay=1e-5)

    return model, lossfun, optimizer
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Zhenghao Li
# Date: 2024-11-08

import torch
import my_net

batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model, lossfun, optimizer = my_net.classify.makeEmotionNet(False)
PATH = "./face_expression.pth"
model.load_state_dict(torch.load(PATH, weights_only=True))
model.to(device)
test_loader, classes = my_net.utility.loadTest("./images/", batch_size)

X, y = next(iter(test_loader))

model.eval()
## Test in one batch
with torch.no_grad():
    X = X.to(device)
    yHat = model(X)

    ##Step 1 Obtain predicted labels
    # new_labels =

    ##Show first 32 predicted labels
    # my_net.utility.imshow_with_labels(X[:batch_size], new_labels[:batch_size], classes)

    # Step 2
    ##Calculate the accuracy for each category prediction, as well as the overall accuracy
    # Print them to the screen.
    ## "happy:xx.xx%, neutral:xx.xx%, sad:xx.xx%, total:xx.xx%"

    # Step 3
    ##Calculate the recall for each category prediction, as well as the overall recall
    # Print them to the screen.
    ## "happy:xx.xx%, neutral:xx.xx%, sad:xx.xx%, total:xx.xx%"

## Get the accuracy and recall in full dataset
##Step 4
for X, y in test_loader:
    pass

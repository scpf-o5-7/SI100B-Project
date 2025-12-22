import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import sys

transform = {
        "training":
    transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "evaluate":
    transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
}


def imshow_with_labels(images, labels, classes):
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15,15))
    axs = axs.flatten()

    for i in range(grid_size**2):
        if i< num_images:
            img = images.data[i].numpy().transpose((1,2,0))
            img = img/2 + .5 # undo normalization
            label = classes[labels[i]]
            axs[i].imshow(img, cmap='gray')
            axs[i].set_title(f'Label: {label}')
            axs[i].axis('off')
        else:
            axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def load_data(path, batch_size, DataType):
    Dataset = ImageFolder(path, transform=transform[DataType])
    if batch_size == 0:
        batch_size = len(Dataset)
    Dataloader = DataLoader(Dataset ,batch_size=batch_size,shuffle=True, drop_last=True)
    return Dataset, Dataloader

def loadTrain(path, batch_size):
    train_path = os.path.join(path, "train")
    trainset, train_loader = load_data(train_path, batch_size, "training")
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # try again
    print('Data shapes (train/test):')
    print( images.data.shape )
    
    # and the range of pixel intensity values
    print('\nData value range:')
    print( (torch.min(images.data),torch.max(images.data)) )
    
    # Show images
    if sys.stdout.isatty():
        imshow_with_labels(images, labels, trainset.classes)
    else:
        print("Not showing images as not in interactive mode.")

    return train_loader, trainset.classes

def loadTest(path, batch_size=0):
    test_path = os.path.join(path, "validation")
    testset, test_loader = load_data(test_path, batch_size, "evaluate")

    return test_loader, testset.classes

def function2trainModel(model, device, train_loader, lossFun, optimizer):
    epochs = 10

    model.to(device)

    # initialize losses
    trainLoss = np.zeros(epochs)
    trainAcc  = np.zeros(epochs)

    for epochi in range(epochs):
        # loop over training data batches
        model.train() # switch to train mode
        batchLoss = []
        batchAcc = []
        for batch_idx, (X,y) in enumerate(train_loader):
            # push data to GPU
            X = X.to(device)
            y = y.to(device)
            # forward pass and loss
            yHat = model(X)
            loss = lossFun(yHat, y)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss and accuracy for this batch
            batchLoss.append(loss.item())
            batchAcc.append(torch.mean((torch.argmax(yHat, dim=1)==y).float()).item())
            print(f"Epoch: {epochi+1}/{epochs}, Batch: {batch_idx}, {batch_idx+1}/{len(train_loader)}")

        # end of batch loop
        # get average losses and accuracies across the batches
        trainLoss[epochi] = np.mean(batchLoss)
        trainAcc[epochi] = 100*np.mean(batchAcc)
    return trainLoss, trainAcc, model


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import sys

transform = {
    "training": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ),
    "evaluate": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ),
}


def imshow_with_labels(images, labels, classes):
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axs = axs.flatten()

    for i in range(grid_size**2):
        if i < num_images:
            img = images.data[i].numpy().transpose((1, 2, 0))
            img = img / 2 + 0.5  # undo normalization
            label = classes[labels[i]]
            axs[i].imshow(img, cmap="gray")
            axs[i].set_title(f"Label: {label}")
            axs[i].axis("off")
        else:
            axs[i].axis("off")
    plt.tight_layout()
    plt.show()


def load_data(path, batch_size, DataType):
    Dataset = ImageFolder(path, transform=transform[DataType])
    if batch_size == 0:
        batch_size = len(Dataset)
    Dataloader = DataLoader(
        Dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return Dataset, Dataloader


def loadTrain(path, batch_size):
    train_path = os.path.join(path, "train")
    trainset, train_loader = load_data(train_path, batch_size, "training")
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # try again
    print("Data shapes (train/test):")
    print(images.data.shape)

    # and the range of pixel intensity values
    print("\nData value range:")
    print((torch.min(images.data), torch.max(images.data)))

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


def function2trainModel(
    model, device, train_loader, lossFun, optimizer, val_loader=None
):
    epochs = 10
    model.to(device)
    trainLoss = np.zeros(epochs)
    trainAcc = np.zeros(epochs)
    valAcc = np.zeros(epochs)

    for epochi in range(epochs):
        model.train()
        batchLoss, batchAcc = [], []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            yHat = model(X)
            loss = lossFun(yHat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batchLoss.append(loss.item())
            batchAcc.append((torch.argmax(yHat, dim=1) == y).float().mean().item())

        trainLoss[epochi] = np.mean(batchLoss)
        trainAcc[epochi] = 100 * np.mean(batchAcc)

        if val_loader is not None:
            model.eval()
            val_batchAcc = []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    yHat_val = model(X_val)
                    val_batchAcc.append(
                        (torch.argmax(yHat_val, dim=1) == y_val).float().mean().item()
                    )
            valAcc[epochi] = 100 * np.mean(val_batchAcc)
            print(
                f"Epoch {epochi+1}: Train Acc = {trainAcc[epochi]:.2f}%, Val Acc = {valAcc[epochi]:.2f}%"
            )
        else:
            print(f"Epoch {epochi+1}: Train Acc = {trainAcc[epochi]:.2f}%")

    return trainLoss, trainAcc, valAcc, model

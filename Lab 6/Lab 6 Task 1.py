import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import sys
from model import SI100FaceNet
import my_net.utility as utility


def plot_training_curves(losses, train_acc, val_acc=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(losses, "b-", linewidth=2, label="Training Loss")
    ax1.set_title("Training Loss Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(train_acc, "g-", linewidth=2, label="Training Accuracy")
    if val_acc is not None:
        ax2.plot(val_acc, "r-", linewidth=2, label="Validation Accuracy")
    ax2.set_title("Accuracy Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_class_accuracy(accuracy_dict, title="Class Accuracy"):
    classes = list(accuracy_dict.keys())
    accuracies = list(accuracy_dict.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, accuracies, color=["skyblue", "lightgreen", "lightcoral"])

    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)

    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("class_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_sample_predictions(model, test_loader, classes, device, num_samples=12):
    model.eval()

    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    if num_samples > len(images):
        num_samples = len(images)

    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = img / 2 + 0.5

        axes[i].imshow(img)

        true_label = classes[labels[i]]
        pred_label = classes[predicted[i]]
        color = "green" if true_label == pred_label else "red"

        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        axes[i].axis("off")

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=300, bbox_inches="tight")
    plt.show()


def analyze_model_performance():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = "./img"
    batch_size = 32

    test_loader, classes = utility.loadTest(data_path, batch_size)
    print(f"Classes: {classes}")

    model_path = "face_expression.pth"
    model = SI100FaceNet(num_classes=3, printtoggle=False)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    plot_sample_predictions(model, test_loader, classes, device)

    model.eval()
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            yHat = model(X)
            _, predicted = torch.max(yHat, 1)

            all_true_labels.extend(y.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(
        all_true_labels, all_predicted_labels, labels=range(len(classes))
    )

    plot_confusion_matrix(cm, classes)

    class_accuracy = {}
    total_samples = len(all_true_labels)

    for i, class_name in enumerate(classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_accuracy[class_name] = precision * 100

    plot_class_accuracy(class_accuracy)

    class_recall = {}
    for i, class_name in enumerate(classes):
        tp = cm[i, i]  # 真正例
        fn = np.sum(cm[i, :]) - tp  # 假负例
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_recall[class_name] = recall * 100

    plot_class_accuracy(class_recall, title="Class Recall")

    total_correct = np.sum(np.diag(cm))
    overall_accuracy = total_correct / total_samples * 100

    weighted_recall = 0
    for i, class_name in enumerate(classes):
        class_weight = np.sum(cm[i, :]) / total_samples
        weighted_recall += class_recall[class_name] * class_weight / 100

    print(f"\n=== 模型性能总结 ===")
    print(f"整体准确率: {overall_accuracy:.2f}%")
    print(f"加权召回率: {weighted_recall*100:.2f}%")

    from sklearn.metrics import classification_report

    print("\n=== 详细分类报告 ===")
    print(
        classification_report(
            all_true_labels, all_predicted_labels, target_names=classes, digits=4
        )
    )

    losses = [0.90, 0.70, 0.61, 0.55, 0.48, 0.41, 0.33, 0.26, 0.19, 0.14]
    train_acc = [53.75, 68.39, 73.27, 76.67, 80.01, 83.81, 87.14, 90.10, 92.88, 95.17]
    plot_training_curves(losses, train_acc)


if __name__ == "__main__":
    analyze_model_performance()

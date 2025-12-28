import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import sys

sys.path.append('../Lab 4')

from model import SI100FaceNet

import my_net.utility as utility

batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "../Lab 4/face_expression.pth"
data_path = "../Lab 4/img/"

model = SI100FaceNet(num_classes=3, printtoggle=False)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.to(device)

test_loader, classes = utility.loadTest(data_path, batch_size)

X, y = next(iter(test_loader))

model.eval()

with torch.no_grad():
    X = X.to(device)
    yHat = model(X)

    _, predicted = torch.max(yHat, 1)
    new_labels = predicted.cpu().numpy()
    true_labels = y.cpu().numpy()

    utility.imshow_with_labels(
        X[:batch_size].cpu(), new_labels[:batch_size], classes
    )

    cm = confusion_matrix(true_labels, new_labels, labels=range(len(classes)))
    print("\nConfusion Matrix (Single Batch):")
    print("     " + " ".join([f"{cls:>6}" for cls in classes]))
    for i, class_name in enumerate(classes):
        print(f"{class_name:>5} {cm[i]}")
    
    class_precision = {}
    total_correct = 0
    total_samples = len(true_labels)

    for i in range(len(classes)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_precision[classes[i]] = precision * 100
        total_correct += tp

    overall_accuracy = total_correct / total_samples * 100

    print("\n=== Step 2 - Single Batch Precision Results ===")
    for class_name in classes:
        print(f"{class_name}: {class_precision[class_name]:.2f}%")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    class_recall = {}
    macro_recall = 0
    weighted_recall = 0

    for i in range(len(classes)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_recall[classes[i]] = recall * 100
        class_weight = np.sum(cm[i, :]) / total_samples
        weighted_recall += recall * class_weight
        macro_recall += recall

    macro_recall = macro_recall / len(classes) * 100
    weighted_recall *= 100

    print("\n=== Step 3 - Single Batch Recall Results ===")
    for class_name in classes:
        print(f"{class_name}: {class_recall[class_name]:.2f}%")
    print(f"Macro Average Recall: {macro_recall:.2f}%")
    print(f"Weighted Average Recall: {weighted_recall:.2f}%")

print("\n=== Step 4 - Full Dataset Evaluation ===")
all_true_labels = []
all_predicted_labels = []

model.eval()
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        yHat = model(X)
        _, predicted = torch.max(yHat, 1)

        all_true_labels.extend(y.cpu().numpy())
        all_predicted_labels.extend(predicted.cpu().numpy())

all_true_labels = np.array(all_true_labels)
all_predicted_labels = np.array(all_predicted_labels)

cm_full = confusion_matrix(all_true_labels, all_predicted_labels, labels=range(len(classes)))
print("Full Dataset Confusion Matrix:")
print("     " + " ".join([f"{cls:>6}" for cls in classes]))
for i, class_name in enumerate(classes):
    print(f"{class_name:>5} {cm_full[i]}")

full_class_precision = {}
full_total_correct = 0
full_total_samples = len(all_true_labels)

for i in range(len(classes)):
    tp = cm_full[i, i]
    fp = np.sum(cm_full[:, i]) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    full_class_precision[classes[i]] = precision * 100
    full_total_correct += tp

full_overall_accuracy = full_total_correct / full_total_samples * 100

print("\n=== Full Dataset Precision Results ===")
for class_name in classes:
    print(f"{class_name}: {full_class_precision[class_name]:.2f}%")
print(f"Overall Accuracy: {full_overall_accuracy:.2f}%")

full_class_recall = {}
full_macro_recall = 0
full_weighted_recall = 0

for i in range(len(classes)):
    tp = cm_full[i, i]
    fn = np.sum(cm_full[i, :]) - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    full_class_recall[classes[i]] = recall * 100
    class_weight = np.sum(cm_full[i, :]) / full_total_samples
    full_weighted_recall += recall * class_weight
    full_macro_recall += recall

full_macro_recall = full_macro_recall / len(classes) * 100
full_weighted_recall *= 100

print("\n=== Full Dataset Recall Results ===")
for class_name in classes:
    print(f"{class_name}: {full_class_recall[class_name]:.2f}%")
print(f"Macro Average Recall: {full_macro_recall:.2f}%")
print(f"Weighted Average Recall: {full_weighted_recall:.2f}%")

print("\n=== Detailed Classification Report ===")
print(classification_report(all_true_labels, all_predicted_labels, target_names=classes, digits=4))
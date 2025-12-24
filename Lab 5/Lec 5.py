import torch
import my_net
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model, lossfun, optimizer = my_net.classify.makeEmotionNet(False)
PATH = "../Lab 4/face_expression.pth"
model.load_state_dict(torch.load(PATH, weights_only=True))
model.to(device)
test_loader, classes = my_net.utility.loadTest("../Lab 4/img/", batch_size)

X, y = next(iter(test_loader))

model.eval()

with torch.no_grad():
    X = X.to(device)
    yHat = model(X)

    _, predicted = torch.max(yHat, 1)
    new_labels = predicted.cpu().numpy()
    true_labels = y.cpu().numpy()

    my_net.utility.imshow_with_labels(
        X[:batch_size].cpu(), new_labels[:batch_size], classes
    )

    cm = confusion_matrix(true_labels, new_labels)
    print("Confusion Matrix:")
    print(cm)

    class_accuracy = {}
    total_correct = 0
    total_samples = len(true_labels)

    for i in range(len(classes)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_accuracy[classes[i]] = precision * 100
        total_correct += tp

    overall_accuracy = total_correct / total_samples * 100

    print("\nStep 2 - Accuracy Results:")
    for class_name in classes:
        print(f"{class_name}: {class_accuracy[class_name]:.2f}%")
    print(f"Total: {overall_accuracy:.2f}%")

    class_recall = {}
    weighted_recall = 0

    for i in range(len(classes)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_recall[classes[i]] = recall * 100
        class_weight = np.sum(cm[i, :]) / total_samples
        weighted_recall += recall * class_weight * 100

    print("\nStep 3 - Recall Results:")
    for class_name in classes:
        print(f"{class_name}: {class_recall[class_name]:.2f}%")
    print(f"Total: {weighted_recall:.2f}%")

print("\nStep 4 - Full Dataset Evaluation:")
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

cm_full = confusion_matrix(all_true_labels, all_predicted_labels)
print("Full Dataset Confusion Matrix:")
print(cm_full)

full_class_accuracy = {}
full_total_correct = 0
full_total_samples = len(all_true_labels)

for i in range(len(classes)):
    tp = cm_full[i, i]
    fp = np.sum(cm_full[:, i]) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    full_class_accuracy[classes[i]] = precision * 100
    full_total_correct += tp

full_overall_accuracy = full_total_correct / full_total_samples * 100

full_class_recall = {}
full_weighted_recall = 0

for i in range(len(classes)):
    tp = cm_full[i, i]
    fn = np.sum(cm_full[i, :]) - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    full_class_recall[classes[i]] = recall * 100
    class_weight = np.sum(cm_full[i, :]) / full_total_samples
    full_weighted_recall += recall * class_weight * 100

print("\nFull Dataset Accuracy Results:")
for class_name in classes:
    print(f"{class_name}: {full_class_accuracy[class_name]:.2f}%")
print(f"Total: {full_overall_accuracy:.2f}%")

print("\nFull Dataset Recall Results:")
for class_name in classes:
    print(f"{class_name}: {full_class_recall[class_name]:.2f}%")
print(f"Total: {full_weighted_recall:.2f}%")

print("\nDetailed Classification Report:")
print(
    classification_report(all_true_labels, all_predicted_labels, target_names=classes)
)

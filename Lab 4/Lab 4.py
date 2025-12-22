import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = '1'

import torch
import my_net

from model import SI100FaceNet
from config import CLASS_CONFIG

task_type = input("Enter task type (basic/bonus): ").strip().lower()
config = CLASS_CONFIG[task_type]

batchsize = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training Device: ", device)

# Load dataset
train_loader, classes= my_net.utility.loadTrain(config['data_path'], batchsize)
val_loader, _ = my_net.utility.loadTest(config['data_path'], batchsize) 

# Set model, lossfunc and optimizer
model = SI100FaceNet(num_classes=config['num_classes'], printtoggle=True)
model = model.to(device)
lossfun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Start training process
losses, train_accuracy, val_accuracy, model = my_net.utility.function2trainModel(
    model, device, train_loader, lossfun, optimizer, val_loader
)

print("--------------------------")
print("Loss and accuracy in every iteration")
for i, (loss, train_acc, val_acc) in enumerate(zip(losses, train_accuracy, val_accuracy)):
    print(f"Iteration {i}, loss: {loss:.4f}, train_accuracy: {train_acc:.2f}%")

PATH = config["save_name"]
torch.save(model.state_dict(), PATH)
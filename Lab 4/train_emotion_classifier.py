import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import my_net

from model import emotionNet

batchsize    = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training Device: ", device)

# Load dataset
train_loader, classes= my_net.utility.loadTrain("./images", batchsize)

# Set model, lossfunc and optimizer
model = emotionNet(printtoggle=True)
model = model.to(device)
lossfun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Start training process
losses, accuracy, _ = my_net.utility.function2trainModel(model, device, train_loader, lossfun, optimizer)

print("--------------------------")
print("Loss and accuracy in every iteration")
for i, (loss, acc) in enumerate(zip(losses, accuracy)):
    print(f"Iteration {i}, loss：{loss:.2f}, accuracy: {acc:.2f}")

PATH = './face_expression.pth'
torch.save(model.state_dict(), PATH)
import torch
from torch import nn 
from dataset import RSNADataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0, resnet50

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train() # set the model to training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.long) # send the instances to the device

        pred = model(X)
        loss = loss_fn((pred), y) # predict and calculate the loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            output.write(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]\n")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()    # set the model in evaluation mode

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.long)
            print(y)
            pred = model(X)
            print(pred)
            l = loss_fn((pred), y)
            print(l)
            test_loss += l.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    output.write(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



trainDataset = RSNADataset("F:\\brain-tumor-target-7\\preprocessed", "F:\\brain-tumor-target-7\\train_labels.csv", "train", 0.118)
valDataset = RSNADataset("F:\\brain-tumor-target-7\\preprocessed", "F:\\brain-tumor-target-7\\train_labels.csv", "val", 0.118)

dloader = DataLoader(trainDataset, batch_size=10, shuffle=True)
val = DataLoader(valDataset, batch_size=10, shuffle=True)

model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)


nig = nn.Sequential(*list(model.children())[:-1])

for layer in nig:
    layer.requires_grad_ = False

testmodel = nn.Sequential(nn.Conv2d(14, 3, kernel_size=(3, 3), padding=(1, 1), bias=False), nig, nn.Flatten(), nn.Linear(1280, 2, bias=True))

device = "cuda" if torch.cuda.is_available() else "cpu"


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 4e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma= 0.01)

testmodel = testmodel.to(device)

output = open("EfficientNet_B1_IMAGENET1K_V1", "w")
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    output.write(f"Epoch {t+1}\n-------------------------------\n")
    train(dloader, testmodel, loss_fn, optimizer)
    test(val, testmodel, loss_fn)
    scheduler.step()
print("Done!")




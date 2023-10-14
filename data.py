import torch
from torch import nn 
from dataset import RSNADataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0, resnet50, inception_v3, densenet121
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict



batchSize = 16
epochs = 60



def train_model(model, weights, size, folder):
    trainDataset = RSNADataset("./brain-tumor-target/preprocessed", "./rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv", "train", 0.118, scanType=folder, modelSize=size)
    valDataset = RSNADataset("./brain-tumor-target/preprocessed", "./rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv", "val", 0.118, scanType=folder, modelSize=size)


    dloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=False)
    val = DataLoader(valDataset, batch_size=batchSize, shuffle=False)

    model.load_state_dict(torch.load(weights), strict=False)

    nig = nn.Sequential(*list(model.children())[:-1])

    layers = list(nig.children())


    for i in range(len(layers) - 2):
        layers[i].requires_grad_ = False


    mergedModel = nn.Sequential(nig, nn.Flatten(), nn.Linear(2048, 2, bias=True), nn.Sigmoid())


    return mergedModel, dloader, val



def train(dataloader, model, loss_fn, optimizer, device, output):
    size = len(dataloader.dataset)

    model.train() # set the model to training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.long) # send the instances to the device

        pred = model(X)
        loss = loss_fn((pred), y) # predict and calculate the loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    #     if batch % 100 == 0:
    #         loss, current = loss.item(), (batch + 1) * len(X)
    #         print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    #         output.write(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]\n")
    
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        output.write(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]\n")
        return

def test(dataloader, model, loss_fn, device, output):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()    # set the model in evaluation mode

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.long)
            #print(y)
            pred = model(X)
            #print(pred)
            l = loss_fn((pred), y)
            #print(l)
            test_loss += l.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    output.write(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


model1, d1, v1 = train_model(resnet50(), "/Users/macbook/Downloads/RadImageNet_pytorch/ResNet50.pt", "s", "FLAIR") # 2048
model2, d2, v2 = train_model(resnet50(), "/Users/macbook/Downloads/RadImageNet_pytorch/ResNet50.pt", "s", "T1w") # 2048
model3, d3, v3 = train_model(resnet50(), "/Users/macbook/Downloads/RadImageNet_pytorch/ResNet50.pt", "s", "T1wCE") # 2048
model4, d4, v4 = train_model(resnet50(), "/Users/macbook/Downloads/RadImageNet_pytorch/ResNet50.pt", "s", "T2w") # 2048
# model2, d2, v2 = train_model(inception_v3(), "/Users/macbook/Downloads/RadImageNet_pytorch/InceptionV3.pt", "l")
# model3, d3, v3 = train_model(densenet121(), "/Users/macbook/Downloads/RadImageNet_pytorch/DenseNet121.pt", "s") # 50176



list = [model1, d1, v1, model2, d2, v2, model3, d3, v3, model4, d4, v4]

names = ["RadImageNet-ResNet50", "RadImageNet-InceptionV3", "RadImageNet-DenseNet121"]

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(device)



for i in range(4):

    model = list[3*i]
    dloader = list[3*i + 1]
    val = list[3*i + 2]

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1000)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma= 0.1)

    output = open(f"./trainingResults/{names[0]}folder{i}", "w")
    epochs = epochs
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        output.write(f"Epoch {t+1}\n-------------------------------\n")
        train(dloader, model, loss_fn, optimizer, device, output)
        test(val, model, loss_fn, device, output)
        scheduler.step()

    torch.save(model.state_dict(), f"./modelWeights/{names[0]}folder{i}.pt")
    print("Done!")
    output.close()




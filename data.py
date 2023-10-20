import torch
from torch import nn 
from dataset import RSNADataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0, resnet50, inception_v3, densenet121
from torchvision.models._api import WeightsEnum
from pathlib import Path
from torch.hub import load_state_dict_from_url
import sys
import ensemble
import time
from tempfile import TemporaryDirectory
import os



""" 
Training and validation module for RSNA Dataset.
Args:
    model: 0 for EfficientNet_B0, 1 for ResNet50, 2 for Inception_V3, 3 for DenseNet121
    weights: The folder of the pretrained models, None if the weights are to be downloaded (only for Pytorch default models)
    dataset: The folder containing the dataset, as passed to the constructor.
    labels: The address of the labels file.
    example:
        python3 data.py 1 ./weights/ResNet50.pt ./brain-tumor-target/preprocessed ./brain-tumor-target/train_labels.csv

"""


'''
This method is necessary in downloading pretrained weights of ResNet50. May prove to be useless
in different platforms.
'''
# def get_state_dict(self, *args, **kwargs):
#     kwargs.pop("check_hash")
#     return load_state_dict_from_url(self.url, *args, **kwargs)
# WeightsEnum.get_state_dict = get_state_dict



batchSize = 16
epochs = 60

models = [efficientnet_b0(), resnet50(), inception_v3(), densenet121()]


# architecture = models[int(sys.argv[0])]
# weights = sys.argv[1]
# trainingFolder = sys.argv[2]
# labelsDirectory = sys.argv[3]
architecture = efficientnet_b0()
weights = EfficientNet_B0_Weights
trainingFolder = Path("F:\\brain-tumor-target\\reorganized")
labelsDirectory = Path.cwd() / "train_labels.csv"

transformations = transforms.Compose((transforms.Resize(224), transforms.Normalize(mean=[0.485,0.456, 0.406], std=[0.229, 0.224, 0.225])))


trainDataset = RSNADataset(trainingFolder, labelsDirectory, "train", 0.118, transform=transformations)
valDataset = RSNADataset(trainingFolder, labelsDirectory, "val", 0.118, transform=transformations)

trainDL = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
valDL = DataLoader(valDataset, batch_size=batchSize, shuffle=True)

dataloaders = {"train": trainDL, "val": valDL}

# model = ensemble.EnsembleModel(architecture, weights)
modelf = efficientnet_b0(EfficientNet_B0_Weights(EfficientNet_B0_Weights.IMAGENET1K_V1))
for layer in modelf.children():
    layer.requires_grad_ = False

model = nn.Sequential(*list(modelf.children())[:-1], nn.LazyLinear(2))

dataset_sizes = [trainDataset.__len__(), valDataset.__len__()]


'''
train and test methods are copied directly from pytorch's getting started page.
There is a huge probability that these need to be modified.
'''

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device, dtype=torch.float32)
                    labels = labels.to(device, dtype = torch.long)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

# def train(dataloader, model, loss_fn, optimizer, device, output):
    size = len(dataloader.dataset)

    model.train() # set the model to training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.long) # send the instances to the device

        pred = model(X)
        loss = loss_fn(pred, y) # predict and calculate the loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            output.write(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]\n")
#     
#     
# 
# def test(dataloader, model, loss_fn, device, output):
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
            l = loss_fn(pred, y)
            #print(l)
            test_loss += l.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.8f}%, Avg loss: {test_loss:>8f} \n")
    output.write(f"Test Error: \n Accuracy: {(100*correct):>0.8f}%, Avg loss: {test_loss:>8f} \n")
# 

# model1, d1, v1 = train_model(resnet50(), "/Users/macbook/Downloads/RadImageNet_pytorch/ResNet50.pt", "s", "FLAIR") # 2048
# model2, d2, v2 = train_model(resnet50(), "/Users/macbook/Downloads/RadImageNet_pytorch/ResNet50.pt", "s", "T1w") # 2048
# model3, d3, v3 = train_model(resnet50(), "/Users/macbook/Downloads/RadImageNet_pytorch/ResNet50.pt", "s", "T1wCE") # 2048
# model4, d4, v4 = train_model(resnet50(), "/Users/macbook/Downloads/RadImageNet_pytorch/ResNet50.pt", "s", "T2w") # 2048
# model2, d2, v2 = train_model(inception_v3(), "/Users/macbook/Downloads/RadImageNet_pytorch/InceptionV3.pt", "l")
# model3, d3, v3 = train_model(densenet121(), "/Users/macbook/Downloads/RadImageNet_pytorch/DenseNet121.pt", "s") # 50176

# the numbers next to the lines denote input features of the architecture for each different model.




# the list to automate training of different models.
# list = [model1, d1, v1, model2, d2, v2, model3, d3, v3, model4, d4, v4]
# 
names = ["EfficientNet", "RadImageNet-ResNet50", "RadImageNet-InceptionV3", "RadImageNet-DenseNet121"]

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(device)


model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.001, momentum=0.87, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma= 0.1)

# output = open(f"./trainingResults/{names[int(sys.argv[0])]}", "w")

train_model(model, loss_fn, optimizer, scheduler, 10)

# epochs = epochs
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     output.write(f"Epoch {t+1}\n-------------------------------\n")
#     train(trainDL, model, loss_fn, optimizer, device, output)
#     test(valDL, model, loss_fn, device, output)
#     # scheduler.step()

torch.save(model.state_dict(), f"./modelWeights/{names[int(sys.argv[0])]}.pt")
print("Done!")
# output.close()




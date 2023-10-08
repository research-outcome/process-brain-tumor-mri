from pathlib import Path
from natsort import natsorted
import torch

def uniform_temporal_subsample(x, num_samples):
    '''
        Moddified from https://github.com/facebookresearch/pytorchvideo/blob/d7874f788bc00a
7badfb4310a912f6e531ffd6d3/pytorchvideo/transforms/functional.py#L19
    '''
    t = len(x)
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return [x[i] for i in indices]

def uniformTemporal(dataList: Path, number: int):

    dataLength = len(dataList)
    interval = dataLength // (number - 1)

    finalList = list()
    j = 0
    for i in range(0, dataLength, interval):
        if j == number:
            break
        finalList.append(dataList[i])
        j += 1
    return finalList

def mixedSampling(dataList, number):

    finalList = list()

    dataLength = len(dataList)
    temp = list()
    middle = dataLength // 2 - 1
    for i in range(middle - 2, middle + 4):
        finalList.append(dataList[i])
        temp.append(dataList[i])
    
    for i in temp: 
        dataList.remove(i)

    del temp

    interval = len(dataList) // (number - 6)
    for i in range(0, len(dataList), interval):
        finalList.append(dataList[i])

    return natsorted(finalList)





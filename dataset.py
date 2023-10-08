import torch
import pathlib
import pydicom
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class RSNADataset(Dataset):

    def __init__(self, root_dir, csv_dir, transform=None, augment=None, randomizer=None, scanType: str = "FLAIR"):
        self.root_dir = root_dir
        self.scanType = scanType
        self.buffers = (0, 0, 0, 0)
        self.transform = transform
        self.randomizer = randomizer

        csv_data = pd.read_csv(csv_dir)
        self.labels = csv_data[csv_data.columns[1]]


    # there definitely exists a better way.
    def __len__(self):
        rootDir = pathlib.Path(self.root_dir)
        return len(list(rootDir.iterdir()))
    
    def setBuffers(self, buffers):
        self.buffers = buffers



    def __getitem__(self, index: int):
        rootDir = pathlib.Path(self.root_dir)

    
        patientList = list(rootDir.iterdir())
        patient = patientList[index]
        flairDir = pathlib.Path(self.root_dir + "\\" + patient.name + "\\" + self.scanType)
        tensor = self.concatenate(list(flairDir.iterdir()))
        
        return (tensor, self.labels[index])


        
    def concatenate(self, dataList: list):

        randomizer = self.randomizer

        tensorsList = list()
        randomizedList = randomizer(dataList) if randomizer != None else dataList

        if type(randomizedList) != list:
            raise TypeError("Invalid randomizer or data, use lists.")

        if len(randomizedList) == 0:
            raise ValueError("The folder is empty")

        try:
            for data in randomizedList:
                dicom = pydicom.dcmread(str(data))
                npdicom = dicom.pixel_array

                if self.transform:
                    npdicom = self.transform(npdicom)

                dicomtensor = torch.from_numpy((npdicom // 256).astype(np.uint8))

                tensorsList.append(dicomtensor)
        except:
            raise TypeError("Invalid data type.")
        
        tensor = torch.stack(tensorsList)
        if self.augment:
            self.augment(tensor)

        return tensor

    def crop(array, buffers: tuple):
        imshape = array.shape
        return array[buffers[0]:(imshape[0] - buffers[1]), buffers[2]: (imshape[1] - buffers[3])]

            

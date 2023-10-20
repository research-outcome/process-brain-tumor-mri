import torch
import pathlib
import pydicom
import numpy as np
import pandas as pd
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from methods import *
import natsort

from torch.utils.data import Dataset


class RSNADataset(Dataset):

    def __init__(self, root_dir, csv_dir=None, dataset="train", val=0, scanType: str = "FLAIR"):
        self.root_dir = root_dir
        self.scanType = scanType
        self.dataset = dataset
        self.val = val

        if csv_dir != None:
            csv_data = pd.read_csv(csv_dir)
            labels = csv_data[csv_data.columns[1]]
            patients = list(csv_data.index)
            self.trainingData, self.valData = self.train_val_generate(labels, patients)
        
        else:
            self.testData = self.test_generate(root_dir)  


    def test_generate(self, path):
        
        testPatients = list()
        testDir = pathlib.Path(path)
        for patient in testDir.iterdir():
            testPatients.append(patient)
        
        return shuffle(testPatients, random_state=42)



    def train_val_generate(self, labels, patients):
        
        liste = list()
        if self.val:
            trainX, valX, _, _ = train_test_split(patients, labels, test_size=self.val, random_state=42)
            liste.append(trainX)
            liste.append(valX)
            trainingSet = liste[0]  
            valSet = liste[1]
        else:
            trainingSet = patients

        trainDataPaths, valDataPaths = list(), list()

        for patient in trainingSet:
            for type in patient.iterdir():
                if type.name == self.scanType:
                    label = labels[patients.index(patient)]
                    for data in type.iterdir():
                        trainDataPaths.append((data, label))
                else:
                    continue

        for patient in valSet:
            for type in patient.iterdir():
                if type.name == self.scanType:
                    label = labels[patients.index(patient)]
                    for data in type.iterdir():
                        valDataPaths.append((data, label))
                else:
                    continue
        
        trainingData = shuffle(trainDataPaths, random_state=42)
        valData = shuffle(valDataPaths, random_state=42)

        return trainingData, valData
            
                


    # there definitely exists a better way.
    def __len__(self):
        if self.dataset == "train":
            return len(self.trainingData)
        elif self.dataset == "val":
            return len(self.valData)
        else:
            return len(self.testData)


    def __getitem__(self, index: int):

        if self.dataset == "train":
            fetchedItem = self.trainingData[index]
            label = fetchedItem[1]
            fetchedData = (np.load(fetchedItem[0])).astype(np.int32)
            return (fetchedData, label)

        elif self.dataset == "val":
            fetchedItem = self.valData[index]
            label = fetchedItem[1]
            fetchedData = (np.load(fetchedItem[0])).astype(np.int32)
            return (fetchedData, label)
        
        else:
            patient = self.testData[index]
            fetchedItem = self.transform(patient)
            if self.modelSize == 's':
                fetchedItem = cv.resize(fetchedItem, (224, 224), interpolation=cv.INTER_LINEAR)
            elif self.modelSize == 'l':
                fetchedItem = cv.resize(fetchedItem, (299, 299), interpolation=cv.INTER_LINEAR)
            return fetchedItem.astype(np.int32)
    
    
    def transform(self, patient):

        subsampled = list()

        for type in patient.iterdir():
            if type.name == self.scanType:
                directory = natsort.natsorted(type.iterdir())

        for file in directory:
            dicom = pydicom.dcmread(str(file))
            npdicom = dicom.pixel_array
            zeros = cv.countNonZero(npdicom)
            if zeros > 100:
                subsampled.append(npdicom)

        if len(subsampled != 0):
            subsampled = uniform_temporal_subsample(subsampled, 11)

            # begin cropping
            croppedList = list()
            for item in subsampled:
                item = Crop(item)
                croppedList.append(item)

            return concatenate(croppedList)
        else:
            return [np.zeros((299, 299)) for i in range(9)]




        


            

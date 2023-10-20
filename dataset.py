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

    def __init__(self, root_dir, csv_dir=None, dataset="train", val=0, scanType: str = "FLAIR", transform = None):
        self.root_dir = root_dir
        self.scanType = scanType
        self.dataset = dataset
        self.val = val
        self.transform = transform
        self.types = ["FLAIR", "T1w", "T1wCE", "T2w"]

        if csv_dir != None:
            csv_data = pd.read_csv(csv_dir, dtype=str)
            labels = csv_data.astype("Int64")[csv_data.columns[1]]
            self.patients = list(csv_data["BraTS21ID"])
            self.trainingData, self.valData = self.train_val_generate(labels)
        
        else:
            self.testData = self.test_generate(root_dir)  


    def test_generate(self, path):
        
        testPatients = list()
        testDir = pathlib.Path(path)
        for patient in testDir.iterdir():
            testPatients.append(patient)
        
        return shuffle(testPatients, random_state=42)



    def train_val_generate(self, labels):
        patients = self.patients

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
            for type in self.types:
                if type == self.scanType:
                    label = labels[patients.index(patient)]
                    file = self.root_dir / type / str(label) 
                    for data in file.iterdir():
                        trainDataPaths.append((data, label))
                else:
                    continue

        for patient in valSet:
            for type in self.types:
                if type == self.scanType:
                    label = labels[patients.index(patient)]
                    file = self.root_dir / type / str(label) 
                    for data in file.iterdir():
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
            if self.transform is not None:
                fetchedData = self.transform(fetchedData)
            return (fetchedData, label)

        elif self.dataset == "val":
            fetchedItem = self.valData[index]
            label = fetchedItem[1]
            fetchedData = (np.load(fetchedItem[0])).astype(np.int32)
            if self.transform is not None:
                fetchedData = self.transform(fetchedData)
            return (fetchedData, label)
        
        else:
            patient = self.testData[index]
            fetchedItem = self.preprocess(patient)
            if self.transform is not None:
                fetchedItem = self.transform(fetchedItem)
            return fetchedItem.astype(np.int32)
    
    
    def preprocess(self, patient):

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




        


            

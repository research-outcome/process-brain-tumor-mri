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
import regex as re

from torch.utils.data import Dataset


class RSNADataset(Dataset):

    def __init__(self, root_dir, csv_dir=None, dataset="train", val=0, modelSize="s"):
        """
        Constructor for the dataset.\n
        Args:
            root_dir: The root directory of the dataset. Must be the directory that is directly parent to the patient folders.
            csv_dir: The directory containing the csv file with the labels.
            dataset: The mode to initialize the dataset in. "train" will return a training set, "val" will return a validation set, "test" will return
            a test set.
            val: The ratio of validation dataset to all dataset.
            scanType: The parametric scan type to initialize the Dataset with. Each dataset is designed to operate on a single parametric scan type.
            modelSize: "s" will return images of size (224, 224), "l" will return images of size (299, 299).
        Returns:
            Dataset object belonging to the specified dataset
        """
        self.root_dir = root_dir
        self.dataset = dataset
        self.val = val
        self.modelSize = modelSize

        if csv_dir != None:
            csv_data = pd.read_csv(csv_dir)
            labels = csv_data[csv_data.columns[1]]
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
        """ 
        Generates the training and validation sets from the patient list.\n
        All augmentations for any given patient is contained within the same set.\n
        Args:
            labels: Ordered dataframe containing labels
        Returns:
            trainingData: Training split of the dataset
            valData: Validation split of the dataset
        """        
        patients = list()

        rootDir = pathlib.Path(self.root_dir)

        for patient in rootDir.iterdir():
            patients.append(patient)

        patients = natsort.natsorted(patients)

        liste = list()
        """
        if val > 0, use train_test_split to split the list containing patient's names
        """
        if self.val:
            trainX, valX, _, _ = train_test_split(patients, labels, test_size=self.val, random_state=42)
            liste.append(trainX)
            liste.append(valX)
            trainingSet = liste[0]  
            valSet = liste[1]
        else:
            trainingSet = patients

        trainDataPaths, valDataPaths = [list(), list(), list(), list()], [list(), list(), list(), list()]

        # for each patient in training split
        for patient in trainingSet:
            # list the subfolders -used to maintain consistency 
            types = natsort.natsorted(patient.iterdir())
            # for each subfolder, retrieve the data and add (data, label) to corresponding list in
            # trainDataPaths
            for i in range(4):
                type = types[i]
                # retrieve the label
                label = labels[patients.index(patient)]
                for data in type.iterdir():
                    # check the size of the data using regex
                    isSize = len(re.findall(f'(?<=-)[{self.modelSize}](?=\.npy)', data.name)) > 0
                    if isSize:
                        trainDataPaths[i].append((data, label))

                else:
                    continue
        # for each patient in validation split, do the same
        for patient in valSet:
            types = natsort.natsorted(patient.iterdir())
            for i in range(4):
                type = types[i]
                label = labels[patients.index(patient)]
                for data in type.iterdir():
                    # checking data size
                    isSize = len(re.findall(f'(?<=-)[{self.modelSize}](?=\.npy)', data.name)) > 0
                    if isSize:
                        valDataPaths[i].append((data, label))
                else:
                    continue
        
        # shuffle the training and validation datasets
        trainingData = [shuffle(trainDataPaths[0], trainDataPaths[1], trainDataPaths[2], trainDataPaths[3], random_state=42)]
        valData = [shuffle(valDataPaths[0], valDataPaths[1], valDataPaths[2], valDataPaths[3], random_state=42)]

        return trainingData, valData
            
                


    def __len__(self):
        if self.dataset == "train":
            return len(self.trainingData[0])
        elif self.dataset == "val":
            return len(self.valData[0])
        else:
            return len(self.testData)


    def __getitem__(self, index: int):

        if self.dataset == "train":
            fetchedItem = [self.trainingData[i][index][0] for i in range(4)]
            label = self.trainingData[0][index][1]
            fetchedData = [(np.load(fetchedItem[i])).astype(np.int32) for i in range(4)]
            return (fetchedData, label)

        elif self.dataset == "val":
            fetchedItem = [self.valData[i][index][0] for i in range(4)]
            label = self.valData[0][index][1]
            fetchedData = [(np.load(fetchedItem[i])).astype(np.int32) for i in range(4)]
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




        


            

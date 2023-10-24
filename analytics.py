#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pydicom
import cv2 as cv
import numpy as np
#import natsort
#import regex as re
import re

import torchvision.transforms as transforms
from torchvision import datasets

trainLocation = "/blue/fl-poly/oguzhan/radimagenet/data/train"



'''
FUNCTION CALLS placed AT THE BOTTOM
'''



# the following one accepts the path.iterdir()
def natsorted(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]  # convert Path to string here
    return sorted(l, key=alphanum_key)


''''
This method calculates the minimum - maximum size of any dicom file within the entire dataset. 
'''
def findNormalizeParameters(location):
    print(location)
    #efficient_b0 size 224
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()])
    dataset = datasets.ImageFolder(root = location, transform=None)
    mean = np.zeros(3)  # Assuming RGB images, change to 1 for grayscale
    std = np.zeros(3)   # Assuming RGB images, change to 1 for grayscale
    for img, _ in dataset:
        img = np.array(img) / 255.0  # Convert pixel values to the range [0, 1]
        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))
    
    mean /= len(dataset)
    std /= len(dataset)
    print(f'Mean: {mean}')
    print(f'Std: {std}')
    print()
    
''''
This method calculates the minimum - maximum size of any dicom file within the entire dataset. 
'''
def sizeCalculate():

    minimumShape = (10000, 10000)
    maximumShape = (0, 0)

    path = Path(trainLocation)
    for patient in path.iterdir():
        print(patient.name)
        for type in patient.iterdir():
            for file in type.iterdir():
                dicom = pydicom.dcmread(str(file))
                npdicom = dicom.pixel_array
                shape = npdicom.shape
                if shape[0] > maximumShape[0]:
                    maximumShape = shape
                if shape[0] < minimumShape[0]:
                    minimumShape = shape
    
    print(minimumShape, maximumShape)

''''
This method calculates number of non-zero pixels within the entire dataset. 
'''
def nonZero():
    originalData = Path('/blue/fl-poly/oguzhan/radimagenet/data/train')

    analyticFile = open("nonZeroCount.txt", "w")

    for patient in originalData.iterdir():
        if not patient.is_dir:
            continue
        print(f"{patient.name} - Beginning")
        for type in patient.iterdir():
            for file in type.iterdir():
                # read the dicom file into a numpy array
                dicom = pydicom.dcmread(str(file))
                npdicom = dicom.pixel_array
                nonZeroCount = cv.countNonZero(npdicom)
                analyticFile.write(f"{patient.name}:{type.name}:{file.name}:{nonZeroCount}\n")
    analyticFile.close()
    

    ''''
This method calculates max number of non-zero pixels for each patient
'''
def nonZeroStat():
    originalData = Path('/blue/fl-poly/oguzhan/radimagenet/data/train')

    analyticFile = open("nonZeroMax.txt", "w")

    for patient in originalData.iterdir():
        if not patient.is_dir:
            continue
        print(f"{patient.name} - Beginning")
        for type in patient.iterdir():
            max = -1
            min = 1000000000
            total = 0
            count = 0
            for file in type.iterdir():
                # read the dicom file into a numpy array
                dicom = pydicom.dcmread(str(file))
                npdicom = dicom.pixel_array
                nonZeroCount = cv.countNonZero(npdicom)
                total = total + nonZeroCount
                count = count + 1
                if (nonZeroCount > max):
                    max = nonZeroCount
                if (nonZeroCount < min):
                    min = nonZeroCount
            analyticFile.write(f"{patient.name}:{type.name}:{max}:{min}:{total/count}:{count}\n")
    analyticFile.close()

def nonZeroNumbers():
    originalData = Path('/blue/fl-poly/oguzhan/radimagenet/data/train')

    analyticFile = open("nonZeroNumbers.txt", "w")

    for patient in originalData.iterdir():
        if not patient.is_dir:
            continue
        print(f"{patient.name} - Beginning")
        for type in patient.iterdir():
            max = -1
            min = 1000000000
            total = 0
            count = 0
            nonZero100 = 0
            nonZero90 = 0
            nonZero80 = 0
            nonZero70 = 0
            nonZero60 = 0
            nonZero50 = 0
            nonZero40 = 0
            for file in type.iterdir():
                # read the dicom file into a numpy array
                dicom = pydicom.dcmread(str(file))
                npdicom = dicom.pixel_array
                nonZeroCount = cv.countNonZero(npdicom)
                total = total + nonZeroCount
                count = count + 1
                if (nonZeroCount > max):
                    max = nonZeroCount
                if (nonZeroCount < min):
                    min = nonZeroCount
                if (nonZeroCount >100):
                    nonZero100 = nonZero100 + 1
                if (nonZeroCount > 90):
                    nonZero90 = nonZero90 + 1
                if (nonZeroCount > 80):
                    nonZero80 = nonZero80 + 1
                if (nonZeroCount > 70):
                    nonZero70 = nonZero70 + 1
                if (nonZeroCount > 60):
                    nonZero60 = nonZero60 + 1
                if (nonZeroCount > 50):
                    nonZero50 = nonZero50 + 1
                if (nonZeroCount > 40):
                    nonZero40 = nonZero40 + 1
            analyticFile.write(f"{patient.name}:{type.name}:{nonZero100}:{nonZero90}:{nonZero80}:{nonZero70}:{nonZero60}:{nonZero50}:{nonZero40}\n")
    analyticFile.close()

''''
This method calculates the number of slices in each given parametric folder, for each patient. 
'''
def sliceHistogram(filePath, folderPath):

    sliceAnalytics = open(filePath, "a")

    folders = ["FLAIR", "T1w", "T1wCE", "T2w"]

    path = Path(folderPath)
    for patient in path.iterdir():
        numbers = [0, 0, 0, 0]
        sliceAnalytics.write(f"{patient.name}:")
        for type in patient.iterdir():
            size = len(list(type.iterdir()))
            index = folders.index(type.name)
            numbers[index] = size
        sliceAnalytics.write(f"{numbers[0]} {numbers[1]} {numbers[2]} {numbers[3]}\n")

    sliceAnalytics.close()


'''
For a given file containing slice numbers for each parametric folder for each patient,
this method computes the minimum number of slices across all patients for each parametric folder.
'''
def findMins(fileList):
    for path in fileList:
        file = open(path, "r")

        min1, min2, min3, min4 = 1000, 1000, 1000, 1000
        for line in file.readlines():
            nums = [int(i) for i in re.findall(r"(?<=[:| ])[0-9+]+", line)]
            if 0 < int(nums[0]) < min1:
                min1 = nums[0]
            if 0 < int(nums[1]) < min2:
                min2 = nums[1]
            if 0 < int(nums[2]) < min3:
                min3 = nums[2]
            if 0 < int(nums[3]) < min4:
                min4 = nums[3]
        
        print(min1, min2, min3, min4)
        file.close()

'''
Given an image, this method computes the maximum contour area for all contours detectable by opencv findContours() method.
'''
def contourArea(nparray):

    blurred = cv.GaussianBlur(nparray, (5, 5), 0)
    thresh = cv.threshold(blurred, 1, 255, cv.THRESH_BINARY)[1]

    close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv.findContours(np.uint8(close), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    maxArea = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > maxArea:
            maxArea = area

    return maxArea


'''
For any parametric folder of each patient, this method will find:
    - the slice with greatest area
    - middle slice
and then print them to an output file.
'''
def middleGreatestComparison(parent):

    analyticFile = open("middle-greatest-comparison.txt", "w")

    parentPath = Path(parent)

    folders = ["FLAIR", "T1w", "T1wCE", "T2w"]

    sfd = 0

    for patient in parentPath.iterdir():

        print(f"{patient.name}: Beginning")

        # if sfd == 2:
        #     analyticFile.close()
        #     return

        data = [[0, 0] for i in range(4)]

        for type in patient.iterdir():

            folderIndex = folders.index(type.name)

            typelist = data[folderIndex]

            index = 0
            targetarea = 0

            for file in natsorted(type.iterdir()):

                dicom = pydicom.dcmread(str(file))
                npdicom = dicom.pixel_array

                area = contourArea(npdicom)
                if area > targetarea:
                    targetarea = area
                    typelist[0] = index
                index += 1

            middle = index // 2

            typelist[1] = middle

        analyticFile.write(f"{patient.name}:({data[0][0]}, {data[0][1]}), ({data[1][0]}, {data[1][1]}), ({data[2][0]}, {data[2][1]}), ({data[3][0]}, {data[3][1]})\n")

        sfd += 1
        print(f"{patient.name}: Done")

    analyticFile.close()


'''
FUNCTION CALLS placed BELOW
'''

findNormalizeParameters("/blue/fl-poly/oguzhan/radimagenet/data/img-reorganized/FLAIR/train")
findNormalizeParameters("/blue/fl-poly/oguzhan/radimagenet/data/img-reorganized/FLAIR/val")
'''
findNormalizeParameters("/blue/fl-poly/oguzhan/radimagenet/data/img-reorganized/FLAIR/test")

findNormalizeParameters("/blue/fl-poly/oguzhan/radimagenet/data/img-reorganized/T1w/train")
findNormalizeParameters("/blue/fl-poly/oguzhan/radimagenet/data/img-reorganized/T1w/val")
findNormalizeParameters("/blue/fl-poly/oguzhan/radimagenet/data/img-reorganized/T1w/test")

findNormalizeParameters("/blue/fl-poly/oguzhan/radimagenet/data/img-reorganized/T1wCE/train")
findNormalizeParameters("/blue/fl-poly/oguzhan/radimagenet/data/img-reorganized/T1wCE/val")
findNormalizeParameters("/blue/fl-poly/oguzhan/radimagenet/data/img-reorganized/T1wCE/test")

findNormalizeParameters("/blue/fl-poly/oguzhan/radimagenet/data/img-reorganized/T2w/train")
findNormalizeParameters("/blue/fl-poly/oguzhan/radimagenet/data/img-reorganized/T2w/val")
findNormalizeParameters("/blue/fl-poly/oguzhan/radimagenet/data/img-reorganized/T2w/test")
'''

'''
FUNCTION CALLS placed ABOVE
'''
from pathlib import Path
import natsort
import numpy as np
import pydicom
import os
import cv2 as cv
from methods import *
import pandas as pd
from matplotlib import pyplot as plt


def augmentations(nparray):
    # first augmentations are 90, 270 rotation
    lastIndex = len(nparray.shape) - 1
    rot90 = np.rot90(nparray, 1, (lastIndex - 1, lastIndex))
    rot270 = np.rot90(nparray, 3, (lastIndex - 1, lastIndex))

    # second set of augmentations are flipping - this needs to be tested
    horizontalFlip = np.flip(nparray, -2)
    verticalFlip = np.flip(nparray, -1)

    # the last set of augmentations are translation
    ltr = translate(-10, 0, nparray)
    rtr = translate(10, 0, nparray)
    utr = translate(0, -10, nparray)
    dtr = translate(0, 10, nparray)

    return [nparray, rot90, rot270, horizontalFlip, verticalFlip, ltr, rtr, utr, dtr]




def transformationPipeline(dataList, analytics):
    # first do the sampling 
    if len(dataList) == 0:
        return [np.zeros((240, 240)) for i in range(9)]
    
    sampled = uniform_temporal_subsample(dataList, 11)

    newList = list()
    # crop each picture, then merge
    for item in sampled:
        dicom = pydicom.dcmread(os.readlink(item))
        npdicom = dicom.pixel_array
        cropped = Crop(npdicom, analytics)
        newList.append(cropped)
    merged = concatenate(newList)

    # now apply the augmentations
    augmentedList = augmentations(merged)
    return augmentedList


def save_fig(img, path, imageName, tight_layout=True, fig_extension="png", resolution=300):
    path.mkdir(parents=True, exist_ok=True)
    imagePath = path / imageName
    if tight_layout:
        plt.tight_layout()
    plt.imshow(img, cmap=plt.cm.bone)
    plt.axis('off')
    plt.savefig(imagePath, format=fig_extension, dpi=resolution)
    plt.close()


def save(parent, name, list, suffixes, as_png=False):
    for i in range(len(list)):
        a1 = np.array([list[i], list[i], list[i]])
        imgName =  f"{name}{suffixes[i]}"
        spath = parent / imgName
        if not as_png:
            np.save(spath, a1)
        else:
            save_fig(a1, spath, imgName)
        



root = Path("F:\\brain-tumor-target\\")
newData = root / "train"
processed = root / "reorganized"
processed.mkdir(parents=True,exist_ok=True)

# fill it
suffixes = ["", "rot90", "rot270", "hf", "vf", "ltr", "rtr", "utr", "dtr"]


csv_dir = Path.cwd() / "train_labels.csv"
csv = pd.read_csv(csv_dir)

patients = csv["BraTS21ID"]
labels = csv["MGMT_value"]


def main():

    i = 0
    histogramDir = Path.cwd() / "analytics" / "resizeHistogram-new.txt"
    histogramFile = open(histogramDir, "w")
    # main loop
    for patient in natsort.natsorted(newData.iterdir()):

        # if i==1:
        #     break

        if not patient.is_dir:
            continue
        
        # create the patient directory in preprocessed
        histogramFile.write(f"{patient.name}:\n")

        types = ["FLAIR", "T1w", "T1wCE", "T2w"]
        for type in types:
            # create the corresponding type directory
            typeTargetFolder = processed / type
            typeTargetFolder.mkdir(parents=True, exist_ok=True)

            histogramFile.write(f"{type.name}:\n")
            print(f"{patient.name} - {type.name}: Beginning")
            # retrieve the label and create directory
            index = list(patients).index(int(patient.name))
            label = labels[index]
            labelDir = typeTargetFolder / str(label)
            labelDir.mkdir(parents=True, exist_ok = True)
            # pass the directory to transformations
            """
            if the patient has no slices for the corresponding scan, decide to fill with zeros or omit entirely
            """
            if (processed / patient.name / type).exists:
                typeArraysList = transformationPipeline(natsort.natsorted(type.iterdir()), histogramFile)
            else:
                typeArraysList = [np.zeros(240, 240) for i in range(9)]
            save(labelDir, patient.name, typeArraysList, suffixes)
            print(f"{patient.name} - {type.name}: Done")

        i+=1
    
    histogramFile.close()


main()


    


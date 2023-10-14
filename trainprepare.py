from pathlib import Path
import randomizer
import natsort
import numpy as np
from pipeline import Crop, translate, concatenate
import pydicom
import os
import cv2 as cv


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
        return [np.zeros((299, 299)) for i in range(9)]
    
    sampled = randomizer.uniform_temporal_subsample(dataList, 11)

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



def save(parent, name, list, suffixes):
    for i in range(len(list)):
        smaller = cv.resize(list[i], (224, 224), interpolation=cv.INTER_LINEAR)
        a1 = np.array([smaller, smaller, smaller])
        larger = cv.resize(list[i], (299, 299), interpolation=cv.INTER_LINEAR)
        a2 = np.array([larger, larger , larger])

        spath = Path(f"{parent}/{name}{suffixes[i]}-s")
        lpath = Path(f"{parent}/{name}{suffixes[i]}-l")
        np.save(spath, a1)
        np.save(lpath, a2)




newData = Path('./brain-tumor-target/train')

continued = True

processed = Path("./brain-tumor-target/preprocessed")
processed.mkdir(parents=True,exist_ok=True)

lastSuccessfulPatient = "00000"

# fill it
suffixes = ["", "rot90", "rot270", "hf", "vf", "ltr", "rtr", "utr", "dtr"]


def main(continued):

    i = 0

    histogramFile = open("./analytics/resizeHistogram-new.txt", "w")
    # main loop
    for patient in natsort.natsorted(newData.iterdir()):

        # if i==1:
        #     break

        if not patient.is_dir:
            continue

        # checkpoint code
        if patient.name != lastSuccessfulPatient:
            if continued:
                continue
        else:
            continued = False
        
        # create the patient directory in preprocessed
        processedPatient = processed / patient.name
        processedPatient.mkdir(parents=True, exist_ok=True)

        histogramFile.write(f"{patient.name}:\n")

        for type in natsort.natsorted(patient.iterdir()):
            histogramFile.write(f"{type.name}:\n")
            print(f"{patient.name} - {type.name}: Beginning")
            prType = processedPatient / type.name
            prType.mkdir(parents=True, exist_ok=True)
            typeArraysList = transformationPipeline(natsort.natsorted(type.iterdir()), histogramFile)
            save(prType, patient.name, typeArraysList, suffixes)
            print(f"{patient.name} - {type.name}: Done")

        i+=1
    
    histogramFile.close()


main(continued)


    


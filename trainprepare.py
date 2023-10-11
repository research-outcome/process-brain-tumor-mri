from pathlib import Path
import randomizer
import natsort
import numpy as np
from pipeline import Crop, translate
import pydicom


def augmentations(nparray):
    # first augmentations are 90, 270 rotation
    lastIndex = len(nparray.shape) - 1
    rot90 = np.rot90(nparray, 1, (lastIndex - 1, lastIndex))
    rot270 = np.rot90(nparray, 3, (lastIndex - 1, lastIndex))

    # second set of augmentations are flipping - this needs to be tested
    horizontalFlip = np.flip(nparray, -2)
    verticalFlip = np.flip(nparray, -1)

    # the last set of augmentations are translation
    ltr = translate(-5, 0, nparray)
    rtr = translate(5, 0, nparray)
    utr = translate(0, -5, nparray)
    dtr = translate(0, 5, nparray)

    return [nparray, rot90, rot270, horizontalFlip, verticalFlip, ltr, rtr, utr, dtr]




def transformationPipeline(dataList, analytics):
    # first do the sampling 
    sampled = randomizer.uniform_temporal_subsample(dataList, 14)

    newList = list()
    # crop each picture, then merge
    for item in sampled:
        dicom = pydicom.dcmread(str(item))
        npdicom = dicom.pixel_array
        cropped = Crop(npdicom, analytics)
        newList.append(cropped)

    merged = np.array(newList)

    # now apply the augmentations
    augmentedList = augmentations(merged)
    return augmentedList



def save(parent, name, list, suffixes):
    for i in range(len(list)):
        path = Path(f"{parent}\\{name}{suffixes[i]}")
        np.save(path, list[i])




newData = Path('F:\\brain-tumor-target-7\\train')

continued = True

processed = Path("F:\\brain-tumor-target-7\\preprocessed\\")
processed.mkdir(parents=True,exist_ok=True)

lastSuccessfulPatient = "00000"

# fill it
suffixes = ["", "rot90", "rot270", "hf", "vf", "ltr", "rtr", "utr", "dtr"]


def main():

    # i = 0

    histogramFile = open(".\\analytics\\resizeHistogram.txt", "w")
    # main loop
    for patient in newData.iterdir():

        # if i==2:
        #     break

        # checkpoint code
        if patient.name != lastSuccessfulPatient:
            if continued:
                continue
        else:
            continued = False
        
        # create the patient directory in preprocessed
        processedPatient = Path(str(processed) + "\\" + patient.name + "\\")
        processedPatient.mkdir(parents=True, exist_ok=True)

        histogramFile.write(f"{patient.name}:\n")

        for type in patient.iterdir():
            histogramFile.write(f"{type.name}:\n")
            print(f"{patient.name} - {type.name}: Beginning")
            prType = Path(str(processedPatient) + "\\" + type.name + "\\")
            prType.mkdir(parents=True, exist_ok=True)
            typeArraysList = transformationPipeline(natsort.natsorted(type.iterdir()), histogramFile)
            save(prType, patient.name, typeArraysList, suffixes)
            print(f"{patient.name} - {type.name}: Done")

        # i+=1
    
    histogramFile.close()


main()


    


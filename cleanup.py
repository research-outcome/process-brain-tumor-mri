import cv2 as cv
import pydicom
from pathlib import Path
import randomizer
import natsort
import numpy as np


pathPrefix = './rsna-miccai-brain-tumor-radiogenomic-classification/train'

# cleanedData = Path('F:\\brain-tumor-target\\train')
originalData = Path('./rsna-miccai-brain-tumor-radiogenomic-classification/train')

newData = Path('./brain-tumor-target/train') 

for patient in originalData.iterdir():
    if not patient.is_dir:
        continue
    print(f"{patient.name} - Beginning")
    for type in patient.iterdir():
        for file in type.iterdir():
            source = file
            newFileFolder = newData / patient.name / type.name
            newFileFolder.mkdir(parents=True, exist_ok=True)
            target = newFileFolder / file.name
            dicom = pydicom.dcmread(str(source))
            npdicom = dicom.pixel_array
            if cv.countNonZero(npdicom) >= 100:
                target.symlink_to(source)
    print(f"{patient.name} - Done")


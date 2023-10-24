#!/usr/bin/env python
# coding: utf-8

''''
The file for cleaning redundant dicom slices from original dataset.

    originalData: location of the unaltered dataset
    example : originalData = Path('./rsna-miccai-brain-tumor-radiogenomic-classification/train')

    newData: location for the new dataset
    newData = Path('./brain-tumor-target/train') 

Note that the program will simply create symbolic links to avoid unnecessary space usage.
'''


import cv2 as cv
import pydicom
from pathlib import Path
import sys
import numpy as np



#originalData = Path(sys.argv[0])
originalData = Path('/blue/fl-poly/oguzhan/radimagenet/data/train')

#newData = Path(sys.argv[1])
newData = Path('/blue/fl-poly/oguzhan/radimagenet/data/cleaned')

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
            # read the dicom file into a numpy array
            dicom = pydicom.dcmread(str(source))
            npdicom = dicom.pixel_array
            # only create a symbolic link if the array has more than given non-zero elements
            if cv.countNonZero(npdicom) >= 100:
                target.symlink_to(source)
    print(f"{patient.name} - Done")


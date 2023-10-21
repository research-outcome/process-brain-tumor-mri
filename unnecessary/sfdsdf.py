from pathlib import Path
import numpy as np
import pydicom

print("hey")

dir = Path("F:\\brain-tumor-target\\train")

max = [(1, 1), (1,1), (1, 1), (1, 1)]
min = [(2000, 2000), (2000,2000), (2000, 2000), (2000, 2000)]

types = ["FLAIR", "T1w", "T1wCE", "T2w"]

for patient in dir.iterdir():
    print(patient.name)
    for type in patient.iterdir():
        for data in type.iterdir():
            dicom = pydicom.dcmread(str(data))
            npdicom = dicom.pixel_array
            size = npdicom.shape
            oldmax = max[types.index(type.name)]
            oldmin = min[types.index(type.name)]
            maxarea = oldmax[0] * oldmax[1]
            minarea = oldmin[0] * oldmin[1]
            area = size[0] * size[1]
            if area < minarea and minarea != 0:
                min[types.index(type.name)] = size
            if area > maxarea and maxarea != 0:
                max[types.index(type.name)] = size

print(max)
print(min)


            
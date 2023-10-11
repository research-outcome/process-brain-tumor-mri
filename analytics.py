from pathlib import Path
import pydicom
import regex as re
import cv2 as cv
import numpy as np
import natsort

def sizeCalculate():

    minimumShape = (10000, 10000)
    maximumShape = (0, 0)

    path = Path("F:\\rsna-miccai-brain-tumor-radiogenomic-classification\\train")
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


def middleGreatestComparison(parent):

    analyticFile = open(".\\analytics\\middle-greatest-comparison.txt", "w")

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

            for file in natsort.natsorted(type.iterdir()):

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

                

# for i in range(5, 8):
#     sliceHistogram(f".\\analytics\\slices-{i}.txt", f"F:\\brain-tumor-target-{i}\\train")

middleGreatestComparison("F:\\brain-tumor-target-5\\train")

# findMins([".\\analytics\\slices-5.txt", ".\\analytics\\slices-6.txt", ".\\analytics\\slices-7.txt"])
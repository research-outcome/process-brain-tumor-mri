'''
    File containing the methods used in image processing.
'''

from pathlib import Path
from natsort import natsorted
import torch
import cv2 as cv
import numpy as np


'''
Taken from https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/280033
Given an array and number of samples, returns a subarray l containing num_samples sampled elements spaced uniformly.
A slight modification is done so that the function will omit the first and last elements of the subarray.
'''
def uniform_temporal_subsample(x, num_samples):
    '''
        Modified from https://github.com/facebookresearch/pytorchvideo/blob/d7874f788bc00a
    7badfb4310a912f6e531ffd6d3/pytorchvideo/transforms/functional.py#L19
    '''
    t = len(x)
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    arr = [x[i] for i in indices]
    return arr[1:len(arr) - 1]


''''
Given a list of contours, returns the one enclosing the greatest area.
'''
def findGreatest(contours):
    area = 0
    v = 0
    for i in range(len(contours)):
        carea = cv.contourArea(contours[i])
        if carea > area:
            area = carea
            v = i
    return contours[v]


'''
Takes an input of list of arrays of size nˆ2, concatenates them such that:
        0, 1, 2, ..., n - 1
        n, n + 1, ..., 2n - 1
        |
        nˆ2 - n, ..., nˆ2 - 1    
 '''
def concatenate(arrays):
    a1 = np.concatenate(arrays[0:3], 1)
    a2 = np.concatenate(arrays[3:6], 1)
    a3 = np.concatenate(arrays[6:], 1)

    arrayfinal = np.concatenate((a1, a2, a3), 0)
    return arrayfinal



''''
Given an image (represented as a numpy array), crops the redundant pixels and resizes the given image.
If analytics is given, the function will write to an output file the resizing done to the important area of 
the image.
'''
def Crop(image, analytics=None):
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    thresh = cv.threshold(blurred, 1, 255, cv.THRESH_BINARY)[1]

    close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv.findContours(np.uint8(close), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    c = findGreatest(contours)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    croppedarray = image[extTop[1]: extBot[1], extLeft[0]: extRight[0]]
    previousShape = croppedarray.shape
    if not analytics:
        analytics.write(f"{previousShape} -> (299, 299)\n")
    resized = cv.resize(croppedarray, (240, 240), interpolation=cv.INTER_LINEAR)

    return resized



''''
Given an image (of type numpy array), translates the image in the x and y direction by tx and ty.
Positive direction corresponds to down and right.
'''
def translate(tx, ty, image):
    N, M = image.shape
    image_translated = np.zeros_like(image)
    image_translated[max(tx,0):M+min(tx,0), max(ty,0):N+min(ty,0)] = image[-min(tx,0):M-max(tx,0), -min(ty,0):N-max(ty,0)]
    return image_translated


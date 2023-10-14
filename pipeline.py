import cv2 as cv
import numpy as np

def findGreatest(contours):
    area = 0
    v = 0
    for i in range(len(contours)):
        carea = cv.contourArea(contours[i])
        if carea > area:
            area = carea
            v = i
    return contours[v]


'''Takes an input of list of arrays of size nˆ2, concatenates them such that:
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

# keep in mind that some of these operations are there to make image prettier.
# also optimize
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

def translate(tx, ty, image):
    N, M = image.shape
    image_translated = np.zeros_like(image)
    image_translated[max(tx,0):M+min(tx,0), max(ty,0):N+min(ty,0)] = image[-min(tx,0):M-max(tx,0), -min(ty,0):N-max(ty,0)]
    return image_translated


    # iimg = cv.merge([image, image, image])
    # ff = cv.normalize(iimg, None, 0, 255, cv.NORM_MINMAX)
    # cnt = cv.drawContours(ff, contours, 0, (255, 0, 0), 5)
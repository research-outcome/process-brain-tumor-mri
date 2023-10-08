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

# keep in mind that some of these operations are there to make image prettier.
# also optimize
def Crop(image):
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    thresh = cv.threshold(blurred, 1, 255, cv.THRESH_BINARY)[1]

    close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=50)

    if (cv.countNonZero(close) > 100):
        contours, hierarchy = cv.findContours(np.uint8(close), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        c = findGreatest(contours)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        croppedarray = image[extTop[1]: extBot[1], extLeft[0]: extRight[0]]
        resized = cv.resize(croppedarray, (224, 224), interpolation=cv.INTER_LINEAR)

    else:
        resized = cv.resize(image, (224, 224), interpolation=cv.INTER_LINEAR)
    return resized


    # iimg = cv.merge([image, image, image])
    # ff = cv.normalize(iimg, None, 0, 255, cv.NORM_MINMAX)
    # cnt = cv.drawContours(ff, contours, 0, (255, 0, 0), 5)
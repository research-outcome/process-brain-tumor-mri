import cv2 as cv
import numpy as np

# keep in mind that some of these operations are there to make image prettier.
# also optimize
def Crop(image: np.ndarray):
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    thresh = cv.threshold(blurred, 254, 255, cv.THRESH_BINARY)[1]

    close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=50)
    contours, _ = cv.findContours(np.uint8(close), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    c = contours[0]
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    croppedarray = image[extLeft: extRight, extTop: extBot]
    return croppedarray


    # iimg = cv.merge([image, image, image])
    # ff = cv.normalize(iimg, None, 0, 255, cv.NORM_MINMAX)
    # cnt = cv.drawContours(ff, contours, 0, (255, 0, 0), 5)
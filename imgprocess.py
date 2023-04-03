import cv2 as cv
import numpy as np

def resizeImg(image, width = None, height = None, inter = cv.INTER_AREA):
    dim = None # Unnecessary?
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    elif height is not None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return (cv.resize(image, dim, interpolation = inter), dim)

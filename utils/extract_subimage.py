import numpy as np
import cv2


def extract_subimage(img, points):
    '''
    Reference: http://www.cocoachina.com/articles/74186
    '''
    mask = np.zeros((img.shape[0], img.shape[1]))

    cv2.fillConvexPoly(mask, points, 1)
    mask = mask.astype(np.bool)

    out = np.zeros_like(img)
    out[mask] = img[mask]

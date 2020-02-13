"""
Library of the various vision processing tools we are using
for the autonomous car project
"""

import cv2 as cv
import numpy as np

def filterNoise(img):
    kernel = np.ones((3,3),np.uint8)

    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    return img

def coneMask(frame):
    """
    find binary mask for cones
    """

    frame = np.copy(frame)
    coneHSVLow = np.array([160,220,220])
    coneHSVHigh = np.array([180,255,255])

    frameCone = cv.inRange(frame, coneHSVLow,coneHSVHigh)
    frameCone = filterNoise(frameCone)
    return frameCone

def wallMask(frame):
    """
    find binary mask for walls
    """

    frame = np.copy(frame)
    wallHSVLow = np.array([80,130,130])
    wallHSVHigh = np.array([100,255,255])

    frameWall = cv.inRange(frame, wallHSVLow,wallHSVHigh)
    frameWall = filterNoise(frameWall)
    return frameWall

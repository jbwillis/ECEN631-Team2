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
    Method by Devin
    """
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    coneHSVLow = np.array([160,220,220])
    coneHSVHigh = np.array([180,255,255])

    frameCone = cv.inRange(frame, coneHSVLow,coneHSVHigh)
    frameCone = filterNoise(frameCone)
    return frameCone

def wallMask(frame):
    """
    find binary mask for walls
    Method by Devin
    """
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    wallHSVLow = np.array([80,130,130])
    wallHSVHigh = np.array([100,255,255])

    frameWall = cv.inRange(frame, wallHSVLow,wallHSVHigh)
    frameWall = filterNoise(frameWall)
    return frameWall

def getConeWallMasks(frame):
    """
    find binary mask for cones and walls using method described
    by Dr. Lee
    """
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    _,s,_ = cv.split(frame_hsv)
    b,g,r = cv.split(frame)
    _, s_thresh = cv.threshold(s, 80, 255, cv.THRESH_BINARY)

    r_thresh = cv.bitwise_and(s_thresh, r)
    _, cones = cv.threshold(r_thresh, 200, 255, cv.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    cones = cv.erode(cones, kernel, iterations = 2)
    cones = cv.dilate(cones, kernel, iterations = 2)
    cones = cv.dilate(cones, kernel, iterations = 2)
    cones = cv.erode(cones, kernel, iterations = 2)

    b_thresh = cv.bitwise_and(s_thresh, b)
    _, wall = cv.threshold(b_thresh, 150, 255, cv.THRESH_BINARY)
    #
    # kernel = np.ones((3,3),np.uint8)
    wall = cv.erode(wall, kernel, iterations = 2)
    wall = cv.dilate(wall, kernel, iterations = 2)
    wall = cv.dilate(wall, kernel, iterations = 2)
    wall = cv.erode(wall, kernel, iterations = 2)
    return cones, wall

def toOccupancyGrid(frame, nx, ny):
    """
    convert a masked frame into an occupancy grid 
    nx: number of horizontal cells
    ny: number of vertical cells
    """

    fx, fy = frame.shape

    og = np.zeros((ny, nx))
    x_step = int(fx/nx+.5) # round up
    y_step = int(fy/ny+.5)
    for ogx_i, fx_i in enumerate(np.arange(0, fx, x_step)):
        for ogy_i, fy_i in enumerate(np.arange(0, fy, y_step)):
            og[ogx_i, ogy_i] = np.count_nonzero(frame[fx_i:fx_i+x_step, fy_i:fy_i+y_step])
    return og

def cropFrame(frame, rm_top=0, rm_bot=0, rm_lef=0, rm_rig=0):
    fx, fy, _ = frame.shape
    return np.copy(frame[rm_top:fx-rm_bot, rm_lef:fy-rm_rig, ...])

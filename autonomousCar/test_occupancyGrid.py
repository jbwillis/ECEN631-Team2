#! /usr/bin/env python3
"""
Test file for testing the occupancy grid creation
"""

import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt

from car_visionTools import *
from car_controlTools import *

import argparse

parser = argparse.ArgumentParser()

gridx, gridy = 10, 10 

plt.ion()
plt.figure(0)

parser.add_argument('in_file', help='the filepath to read the video from')
args = parser.parse_args()

video = cv.VideoCapture(args.in_file); # get video

if video.isOpened():
    ret, frame = video.read()
else:
    ret = False

decisionGrid,x,y = decisionGridGaussian(gridx, gridy, sigx=1., sigy=2., gain=10)
plt_decisionGrid(decisionGrid, x, y, block=False)
while ret:

    start = time.time()
    if cv.waitKey(1) == 27:
        print('break')
        break

    ret, frame0 = video.read()

    frame0 = cropFrame(frame0, rm_bot=20)

    cones, walls = getConeWallMasks(frame0)

    og = toOccupancyGrid(cones+walls, gridx, gridy)

    og = np.clip(og, 0., 1.) # binarize the occupancy grid
    decision = og*decisionGrid
    print(decision)
    mind = np.amin(decision)
    maxd = np.amax(decision)
    if abs(mind) > maxd:
        used = mind
    else:
        used = maxd

    cv.imshow('original', frame0)
    cv.imshow('mask', cones+walls)
    # cv.imshow('og', og)
    plt.figure(0)
    plt.clf()
    plt.pcolormesh(og)
    plt.plot([gridx/2, gridx/2-used], [gridy, gridy/2], lw=2, c='r')
    plt.axis([0, gridx, gridy, 0])
    plt.draw()
    plt.show(block=False)
    plt.pause(.001)
    
    cv.waitKey(30)

video.release()
cv.destroyAllWindows()


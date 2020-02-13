#! /usr/bin/env python3
"""
Test file for testing the occupancy grid creation
"""

import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt

from car_visionTools import toOccupancyGrid, getConeWallMasks

import argparse

parser = argparse.ArgumentParser()

plt.ion()
plt.figure(0)
plt.gca().invert_yaxis()

parser.add_argument('in_file', help='the filepath to read the video from')
args = parser.parse_args()

video = cv.VideoCapture(args.in_file); # get video

if video.isOpened():
    ret, frame = video.read()
else:
    ret = False

while ret:

    start = time.time()
    if cv.waitKey(1) == 27:
        print('break')
        break

    ret, frame0 = video.read()

    cones, walls = getConeWallMasks(frame0)

    og = toOccupancyGrid(cones+walls, 20, 20)

    cv.imshow('original', frame0)
    cv.imshow('mask', cones+walls)
    print(og)
    # cv.imshow('og', og)
    plt.figure(0)
    plt.pcolormesh(og)
    plt.draw()
    plt.show(block=False)
    plt.pause(.001)
    
    cv.waitKey(30)

video.release()
cv.destroyAllWindows()


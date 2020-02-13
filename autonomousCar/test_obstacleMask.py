#! /usr/bin/env python3
"""
Test file for testing the obstacle masking functions
"""

import cv2 as cv
import numpy as np
import time

from car_visionTools import coneMask, wallMask, getConeWallMasks

import argparse

parser = argparse.ArgumentParser()


parser.add_argument('in_file', help='the filepath to read the video from')
args = parser.parse_args()


video = cv.VideoCapture(args.in_file); # get video
# video = cv.VideoCapture('car2.avi'); # get video

if video.isOpened():
    ret, frame = video.read()
else:
    ret = False

# out = cv.VideoWriter('VideoOut.mp4', -1, 20.0, (640,480))

# VOut = cv.VideoWriter()
# VOut.open("VideoOut.avi", -1 , 30, Size(640, 480), 1)


while ret:

    start = time.time()
    if cv.waitKey(1) == 27:
        print('break')
        break

    ret, frame0 = video.read()


    frameCone = coneMask(frame0)
    frameWall = wallMask(frame0)

    cones, walls = getConeWallMasks(frame0)
    
    #total
    total1 = frameWall + frameCone
    total2 = cones + walls

    cv.imshow('original', frame0)
    cv.imshow('total1',total1)
    cv.imshow('total2',total2)

    res1= cv.bitwise_and(frame0,frame0, mask=total1)
    cv.imshow('masked1',res1)

    res2 = cv.bitwise_and(frame0,frame0, mask=total2)
    cv.imshow('masked2',res2)

    print(time.time() - start)

video.release()

# out.release()

cv.waitKey(0)
cv.destroyAllWindows()

input('press enter to exit')

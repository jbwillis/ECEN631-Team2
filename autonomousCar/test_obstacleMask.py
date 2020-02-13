#! /usr/bin/env python3
"""
Test file for testing the obstacle masking functions
"""

import cv2 as cv
import numpy as np

from car_visionTools import coneMask, wallMask

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

    if cv.waitKey(1) == 27:
        print('break')
        break


    if cv.waitKey(1) != -1:
        processType = cv.waitKey(0)


    ret, frame0 = video.read()

    frame = cv.cvtColor(frame0, cv.COLOR_BGR2HSV)

    frameCone = coneMask(frame)
    frameWall = wallMask(frame)
    
    #total
    total = frameWall + frameCone

    cv.imshow('original', frame0)
    # cv.imshow('wall', frameWwall)
    # cv.imshow('cones',frameCone)
    cv.imshow('all',total)

    res = cv.bitwise_and(frame0,frame0, mask= total)


    cv.imshow('best',res)


video.release()

# out.release()

cv.waitKey(0)
cv.destroyAllWindows()

input('press enter to exit')

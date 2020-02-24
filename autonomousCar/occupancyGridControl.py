#! /usr/bin/env python3
"""
Control the car using dr. lee's occupancy grid method
"""

import cv2 as cv
import pyrealsense2 as rs
import numpy as np
import serial
import time
from matplotlib import pyplot as plt

from car_visionTools import *
from car_controlTools import *

driveSpeed = 0.5

gridx, gridy = 10, 10 
decisionGrid,_,_ = decisionGridGaussian(gridx, gridy, sigx=1., sigy=2., gain=10)

cc = carControl()
print("car initialized")

vs = cv.VideoCapture("/dev/video2", cv.CAP_V4L)  # ls -ltr /dev/video*
writer = None
(W, H) = (None, None)

# configure depth stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

print("streaming started")
start = time.time()
while True:

    if start - time.time() > 10:
        break

    (grabbed, frame0) = vs.read()
    if not grabbed:
        break

    frame0 = cropFrame(frame0, rm_bot=20)

    cones, walls = getConeWallMasks(frame0)

    og = toOccupancyGrid(cones+walls, gridx, gridy)

    og = np.clip(og, 0., 1.) # binarize the occupancy grid
    decision = og*decisionGrid
    mind = np.amin(decision)
    maxd = np.amax(decision)
    if abs(mind) > maxd:
        used = mind
    else:
        used = maxd

    used = -used
    print(used)
    cc.drive(driveSpeed)
    cc.steer(used)
    
    cv.imshow('cones', cones)
    cv.imshow('walls', walls)

    cv.waitKey(1)

vs.release()

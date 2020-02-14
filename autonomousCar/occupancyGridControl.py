#! /usr/bin/env python3
"""
Control the car using dr. lee's occupancy grid method
"""

import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt

from car_visionTools import *
from car_controlTools import *

driveSpeed = .5

gridx, gridy = 10, 10 
decisionGrid,_,_ = decisionGridGaussian(gridx, gridy, sigx=1., sigy=2., gain=10)

cc = carControl()

vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)  # ls -ltr /dev/video*
writer = None
(W, H) = (None, None)

# configure depth stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

start = time.time()
while ret:

    if start - time.time() > 30:
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

    print(used)
    cc.drive(driveSpeed)
    cc.steer(used)


vs.release()

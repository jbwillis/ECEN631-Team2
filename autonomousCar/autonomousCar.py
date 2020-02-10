
import cv2 as cv
import numpy as np


def filterNoise(img):
    kernel = np.ones((3,3),np.uint8)

    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    return img

def noChange(frame):
    return frame



####### Task 2


# video = cv.VideoCapture('car.avi'); # get video
video = cv.VideoCapture('car2.avi'); # get video

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

    # cones
    coneHSVLow = np.array([160,220,220])
    coneHSVHigh = np.array([180,255,255])

    frameCone = cv.inRange(frame, coneHSVLow,coneHSVHigh)
    frameCone = filterNoise(frameCone)

    #walls
    wallHSVLow = np.array([80,130,130])
    wallHSVHigh = np.array([100,255,255])

    frameWall = cv.inRange(frame, wallHSVLow,wallHSVHigh)
    frameWall = filterNoise(frameWall)


    #total
    total = frameWall + frameCone

    cv.imshow('original', frame0)
    # cv.imshow('wall', frameWwall)
    # cv.imshow('cones',frameCone)
    # cv.imshow('all',total)

    res = cv.bitwise_and(frame0,frame0, mask= total)


    cv.imshow('best',res)


video.release()

# out.release()

cv.waitKey(0)
cv.destroyAllWindows()

input('press enter to exit')
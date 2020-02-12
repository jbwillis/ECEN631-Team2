import cv2 as cv
import numpy as np

# patternSize = (6,8)

# video = cv.VideoCapture('car_vids/for_transform.avi'); 
patternSize = (7,9)
video = cv.VideoCapture('car_vids/transform.avi'); 

if video.isOpened():
    ret, frame = video.read()
else:
    ret = False

frame_cnt = 0
while ret and frame_cnt < 118:

    if cv.waitKey(1) == 27:
        print('break')
        break 

    
    if cv.waitKey(1) != -1:
        processType = cv.waitKey(0)


    ret, frame0 = video.read()
    frame_cnt += 1

if ret:
    retval, corners = cv.findChessboardCorners(frame0, patternSize)
    if not retval:
        print("Failed to find corners, frame = ", frame_cnt)
    else:
        print("Found corners, frame = ", frame_cnt)
        cv.waitKey()

    # frame_cb = cv.drawChessboardCorners(frame0, patternSize, corners, retval)

    # import pdb; pdb.set_trace()
    # find four outside corners
    out_corners = []
    out_corners.append(corners[0])
    out_corners.append(corners[patternSize[0] -1])
    out_corners.append(corners[patternSize[0]*patternSize[1] - patternSize[0]])
    out_corners.append(corners[patternSize[0]*patternSize[1] - 1])

    for oc in out_corners:
        frame_oc = cv.drawMarker(frame0, tuple(oc[0]), (0, 255, 0))


    cv.imshow('outer corners', frame_oc)
    cv.waitKey()


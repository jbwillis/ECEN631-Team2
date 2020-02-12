import cv2 as cv
import numpy as np
import time

def filterNoise(img):
    kernel = np.ones((3,3),np.uint8)

    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    return img


def main():

    video = cv.VideoCapture('car_vids/car.avi'); # get video
    # video = cv.VideoCapture('car2.avi'); # get video

    if video.isOpened():
        ret, frame = video.read()
    else:
        ret = False

    print(frame.shape)

    # out = cv.VideoWriter('VideoOut.mp4', -1, 20.0, (640,480))

    # VOut = cv.VideoWriter()
    # VOut.open("VideoOut.avi", -1 , 30, Size(640, 480), 1)


    while ret:
        start = time.time()
        if cv.waitKey(1) == 27:
            print('break')
            break


        if cv.waitKey(1) != -1:
            processType = cv.waitKey(0)


        ret, frame = video.read()

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
        _, wall = cv.threshold(b_thresh, 180, 255, cv.THRESH_BINARY)
        #
        # kernel = np.ones((3,3),np.uint8)
        wall = cv.erode(wall, kernel, iterations = 2)
        wall = cv.dilate(wall, kernel, iterations = 2)
        wall = cv.dilate(wall, kernel, iterations = 2)
        wall = cv.erode(wall, kernel, iterations = 2)

        # cones
        coneHSVLow = np.array([160,220,220])
        coneHSVHigh = np.array([180,255,255])

        frameCone = cv.inRange(frame_hsv, coneHSVLow,coneHSVHigh)
        frameCone = filterNoise(frameCone)

        #walls
        wallHSVLow = np.array([80,130,130])
        wallHSVHigh = np.array([100,255,255])

        frameWall = cv.inRange(frame_hsv, wallHSVLow,wallHSVHigh)
        frameWall = filterNoise(frameWall)


        #total
        total = frameWall + frameCone

        cv.imshow('original', frame)
        cv.imshow('b', cones)
        cv.imshow('bf', frameCone)
        # cv.imshow('cones',frameCone)
        # cv.imshow('all',total)

        print(time.time() - start)



    video.release()

    # out.release()

    cv.waitKey(0)
    cv.destroyAllWindows()

    input('press enter to exit')

if __name__=="__main__":
    main()

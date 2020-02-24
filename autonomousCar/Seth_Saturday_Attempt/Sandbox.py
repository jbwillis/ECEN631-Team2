import numpy as np
import cv2
import pyrealsense2 as rs
from car_visionTools import getConeWallMasks, cropFrame, toOccupancyGrid
from car_controlTools import carControl

def main():
    car = carControl()
    default_speed = 0.5

    vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)
    nx = 20
    ny = 20

    turnMat = np.zeros((nx,ny))
    for i in range(0,ny):
        for j in range(0,nx):
            turnMat[i,j] = j;
    
    turnMat[0:nx,int(ny/2):ny] = turnMat[0:nx,int(ny/2):ny] - (ny - 1)
    turnMat = turnMat * 2

    for i in range(0,ny):
        for j in range(0,nx):
            turnMat[i,j] = turnMat[i,j] * ((i+1)/ny)
    turnMat = turnMat + 0.25
    turnMat[ny-3:ny,0:nx] = 0.

    oldCommands = [0.0, 0.0, 0.0, 0.0, 0.0]

    while True:
        ret, frame = vs.read()
        [cones2, walls2] = getConeWallMasks(frame)
        obs = cones2+walls2
        grid = toOccupancyGrid(obs,nx,ny)
        grid = np.array(grid)
        gridThresh = grid
        gridThresh[gridThresh > 0.] = 1
        
        decMat = np.multiply(gridThresh,turnMat)
        argMaxInd = np.unravel_index(np.argmax(np.absolute(decMat)),decMat.shape)
        steerAngle = decMat[argMaxInd] - 0.25
        
        for l in range(4):
            oldCommands[l] = oldCommands[l+1]
        oldCommands[4] = steerAngle
        #steerAngle = 0.0

        car.drive(default_speed)
        car.steer(np.average(oldCommands))

        print(steerAngle)

        frameOut = obs
        #cv2.imshow("Display", frameOut)


        #key = cv2.waitKey(1)
        #if key == ord("q"):
            #break
    vs.release()





if __name__=="__main__":
    main()

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from car_visionTools import getConeWallMasks



def main():

    vid = cv.VideoCapture("rgbd/color_long.avi")
    with open('rgbd/depth_long.npy', 'rb') as f:
        depth_list = np.load(f)

    index = 0
    ret, frame = vid.read()
    depth = depth_list[index]

    while ret:

        cones, walls = getConeWallMasks(frame)

        cone_depth = cv.bitwise_and(depth, depth, mask=cones)
        wall_depth = cv.bitwise_and(depth, depth, mask=walls)

        cv.imshow('cones', cones)
        cv.imshow('walls', walls)
        cv.imshow('depth', (depth/np.max(depth)*255).astype(np.uint8))
        cv.imshow('cdepth', (cone_depth/np.max(cone_depth)*255).astype(np.uint8))
        cv.imshow('wdepth', (wall_depth/np.max(wall_depth)*255).astype(np.uint8))

        cv.waitKey(0)

        # plt.imshow(depth)
        # plt.show()

        index += 1
        ret, frame = vid.read()
        depth = depth_list[index]

    cv.destroyAllWindows()








if __name__=="__main__":
    main()

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
    image_width = depth.shape[1]

    while ret:

        cones, walls = getConeWallMasks(frame)

        cone_depth = cv.bitwise_and(depth, depth, mask=cones)
        wall_depth = cv.bitwise_and(depth, depth, mask=walls)

        num_samples = 300
        # wall_pix = cv.findNonZero(walls)
        wall_pix = np.argwhere(walls != 0)
        if wall_pix.size != 0:
            np.random.shuffle(wall_pix)
            wall_pts = wall_pix[:num_samples,:]
            wall_pts_r = depth[wall_pts[:,0],wall_pts[:,1]]
            wall_pts_b = (wall_pts[:,1] - image_width/2.0)/image_width * np.pi/4
            wall_pts_x = wall_pts_r * np.sin(wall_pts_b)
            wall_pts_y = wall_pts_r * np.cos(wall_pts_b)
            plt.scatter(wall_pts_x, wall_pts_y)
            plt.show()

            # wall_pts_depth = depth[]

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

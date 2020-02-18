import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from car_visionTools import getConeWallMasks


def sat(x, upper, lower):
    if x > upper:
        x = upper
    elif x < lower:
        x = lower
    return x

def get_avoid_mag(norm):
    return sat(1000/norm, 200, 0)


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

        kernel = np.ones((5,5),np.uint8)
        walls = cv.erode(walls, kernel, iterations = 3)

        cone_depth = cv.bitwise_and(depth, depth, mask=cones)
        wall_depth = cv.bitwise_and(depth, depth, mask=walls)

        num_samples = 200
        num_samples_cone = 100
        # wall_pix = cv.findNonZero(walls)
        wall_pix = np.argwhere(wall_depth != 0)
        if wall_pix.size != 0:
            np.random.shuffle(wall_pix)
            wall_pts = wall_pix[:num_samples,:]
            wall_pts_r = depth[wall_pts[:,0],wall_pts[:,1]]
            wall_pts_b = (wall_pts[:,1] - image_width/2.0)/image_width * np.radians(94/2)
            wall_pts_b = wall_pts_b[np.bitwise_and(wall_pts_r > 300, wall_pts_r < 2000)]
            wall_pts_r = wall_pts_r[np.bitwise_and(wall_pts_r > 300, wall_pts_r < 2000)]
            wall_pts_x = wall_pts_r * np.sin(wall_pts_b)
            wall_pts_y = wall_pts_r * np.cos(wall_pts_b)

            cone_pix = np.argwhere(cone_depth != 0)
            if cone_pix.size != 0:
                np.random.shuffle(cone_pix)
                cone_pts = cone_pix[:num_samples_cone,:]
                cone_pts_r = depth[cone_pts[:,0],cone_pts[:,1]]
                cone_pts_b = (cone_pts[:,1] - image_width/2.0)/image_width * np.radians(94/2)
                cone_pts_b = cone_pts_b[np.bitwise_and(cone_pts_r > 300, cone_pts_r < 2000)]
                cone_pts_r = cone_pts_r[np.bitwise_and(cone_pts_r > 300, cone_pts_r < 2000)]
                cone_pts_x = cone_pts_r * np.sin(cone_pts_b)
                cone_pts_y = cone_pts_r * np.cos(cone_pts_b)

                plt.scatter(cone_pts_x, cone_pts_y)

            obstacles = np.concatenate((np.concatenate((np.array([wall_pts_x]),np.array([wall_pts_y])), axis=0),
                                        np.concatenate((np.array([cone_pts_x]),np.array([cone_pts_y])), axis=0)),
                                        axis=1).T

            go_vec = np.array([0.0, 1500.0])
            for i in range(obstacles.shape[0]):
                dist = np.linalg.norm(obstacles[i,:])
                dir = -obstacles[i,:]/dist
                mag = get_avoid_mag(dist)
                go_vec += mag*dir

            go_vec[0] *= 3 #kp for steering, kinda
            print(go_vec)

            plt.arrow(0.0, 0.0, go_vec.item(0), go_vec.item(1))

            # poly = np.polyfit(wall_pts_x, wall_pts_y, 3)
            # x = np.linspace(-500, 500, 100)
            # y = poly[0]*x**3 + poly[1]*x**2 + poly[2]*x + poly[3]
            plt.scatter(wall_pts_x, wall_pts_y)
            # plt.scatter(x,y)
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

import numpy as np
import cv2 as cv
import pyrealsense2 as rs
from car_visionTools import getConeWallMasks, cropFrame
from car_controlTools import carControl

def sat(x, upper, lower):
    if x > upper:
        x = upper
    elif x < lower:
        x = lower
    return x

def get_avoid_mag(norm):
    return sat(50/norm**(0.5), 200, 0)


def main():

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    image_width = 640

    car = carControl()
    default_speed = 0.5
    default_steer = 0.0

    num_samples_wall = 200
    num_samples_cone = 100

    kp = 0.3
    alpha = 0.2
    steer_last = 0.0

    try:

        while True:

            rsframes = pipeline.wait_for_frames()
            aligned_frames = align.process(rsframes)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                raise RuntimeError("Could not acquire depth or color frames.")

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data()) #is this bgr?
            depth_image = cropFrame(depth_image, rm_bot=20)
            color_image = cropFrame(color_image, rm_bot=20)

            cones, walls = getConeWallMasks(color_image)

            kernel = np.ones((5,5),np.uint8)
            walls = cv.erode(walls, kernel, iterations = 2)

            cone_depth = cv.bitwise_and(depth_image, depth_image, mask=cones)
            wall_depth = cv.bitwise_and(depth_image, depth_image, mask=walls)

            wall_pix = np.argwhere(wall_depth != 0)
            if wall_pix.size != 0:
                np.random.shuffle(wall_pix)
                wall_pts = wall_pix[:num_samples_wall,:]
                wall_pts_r = depth_image[wall_pts[:,0],wall_pts[:,1]]
                wall_pts_b = (wall_pts[:,1] - image_width/2.0)/image_width * np.radians(94/2)
                wall_pts_b = wall_pts_b[np.bitwise_and(wall_pts_r > 100, wall_pts_r < 1000)]
                wall_pts_r = wall_pts_r[np.bitwise_and(wall_pts_r > 100, wall_pts_r < 1000)]
                wall_pts_x = wall_pts_r * np.sin(wall_pts_b)
                wall_pts_y = wall_pts_r * np.cos(wall_pts_b)

                cone_pix = np.argwhere(cone_depth != 0)
                if cone_pix.size != 0:
                    np.random.shuffle(cone_pix)
                    cone_pts = cone_pix[:num_samples_cone,:]
                    cone_pts_r = depth_image[cone_pts[:,0],cone_pts[:,1]]
                    cone_pts_b = (cone_pts[:,1] - image_width/2.0)/image_width * np.radians(94/2)
                    cone_pts_b = cone_pts_b[np.bitwise_and(cone_pts_r > 100, cone_pts_r < 2000)]
                    cone_pts_r = cone_pts_r[np.bitwise_and(cone_pts_r > 100, cone_pts_r < 2000)]
                    cone_pts_x = cone_pts_r * np.sin(cone_pts_b)
                    cone_pts_y = cone_pts_r * np.cos(cone_pts_b)

                    obstacles = np.concatenate((np.concatenate((np.array([wall_pts_x]),np.array([wall_pts_y])), axis=0),
                                                np.concatenate((np.array([cone_pts_x]),np.array([cone_pts_y])), axis=0)),
                                                axis=1).T

                else:
                    obstacles = np.concatenate((np.array([wall_pts_x]),np.array([wall_pts_y])), axis=0).T

                go_vec = np.array([0.0, 0.0])
                for i in range(obstacles.shape[0]):
                    dist = np.linalg.norm(obstacles[i,:])
                    direct = -obstacles[i,:]/dist
                    mag = get_avoid_mag(dist)
                    go_vec += mag*direct

                print(go_vec)
                #steer_ang = np.degrees(np.arctan2(go_vec.item(0), go_vec.item(1)))
                #steer_ang = sat(kp * steer_ang, 30, -30)
                steer_ang = sat(kp * go_vec.item(0), 30.0, -30.0)

            else:
                steer_ang = 0.0

            steer_ang = alpha*steer_ang + (1.0-alpha)*steer_last
            car.drive(default_speed)
            car.steer(steer_ang)

            steer_last = steer_ang
            print(steer_ang)
            

            #cv.imshow('cones', cones)
            #cv.imshow('walls', walls)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break


    finally:
        pipeline.stop()

if __name__=="__main__":
    main()

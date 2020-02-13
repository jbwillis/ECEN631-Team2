import numpy as np
import cv2 as cv
import pyrealsense2 as rs

def main():

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    color_out = cv.VideoWriter('color_long.avi',fourcc, 30.0, (640,480))
    #depth_out = cv.VideoWriter('depth.avi',fourcc, 30.0, (640,480))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    depth_list = []

    num_frames = 0
    try:

        while num_frames < 30*2:

            rsframes = pipeline.wait_for_frames()
            aligned_frames = align.process(rsframes)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            #depth_image = np.array([depth_image,depth_image,depth_image])
            #print(depth_image.shape, color_image.shape)
            
            #depth_image = cv.cvtColor(depth_image, cv.COLOR_GRAY2BGR)
            #print(depth_image.shape)
            if not aligned_depth_frame or not color_frame:
                raise RuntimeError("Could not acquire depth or color frames.")


            color_out.write(color_image)
            #depth_out.write(depth_image)
            depth_list.append(depth_image.copy())

            num_frames += 1

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        color_out.release()
        #depth_out.release()
        with open('depth_long.npy', 'wb') as f:
            np.save(f, depth_list)

    finally:
        pipeline.stop()

if __name__=="__main__":
    main()

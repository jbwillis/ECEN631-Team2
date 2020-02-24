import numpy as np
import cv2 as cv
import pyrealsense2 as rs

def main():

    fourcc = cv.VideoWriter_fourcc(*'XVID')

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    num_frames = 0
    try:

        while True:
            rsframes = pipeline.wait_for_frames()
            aligned_frames = align.process(rsframes)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if not aligned_depth_frame or not color_frame:
                raise RuntimeError("Could not acquire depth or color frames.")


            cv.imshow("color", color_image)
            cv.imshow("depth", depth_image)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break


    finally:
        pipeline.stop()

if __name__=="__main__":
    main()

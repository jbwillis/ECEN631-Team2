# USAGE: python3 basic_car_example.py

import pyrealsense2 as rs
import numpy as np
import serial
import time
import cv2


def drive(speed):
    forward_command = "!speed" + str(speed) + "\n"
    ser.write(forward_command.encode())

def steer(degree):
    steer_command = "!steering" + str(degree) + "\n"
    ser.write(steer_command.encode())

# initialize communication with the arduino
#ser = serial.Serial("/dev/ttyUSB0", 115200)
#ser.flushInput()
#time.sleep(2)

# initialize the video stream, pointer to output video file, and frame dimensions
vs = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)  # ls -ltr /dev/video*
writer = None
(W, H) = (None, None)

# configure depth stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Some of these numbers will be different for every car
#ser.write("!start1570\n".encode())
#ser.write("!speed0.0\n".encode())
#ser.write("!inits0.8\n".encode())
#ser.write("!straight1575\n".encode())
#ser.write("!kp0.01\n".encode())
#ser.write("!kd0.01\n".encode())
#ser.write("!pid1\n".encode())

# Required
#drive(1.0)
#time.sleep(.5)
#drive(0)

print("I did things")


while True:

########### Using the Camera ############
    #### RGB ####
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    # UNCOMMENT TO SEE RGB FRAME
    #cv2.imshow("frame", frame)
    #key = cv2.waitKey(1) & 0xFF
    #if key == ord("q"):
    #    break

    #### Depth ####
    # start realsense pipeline
    rsframes = pipeline.wait_for_frames()

    for rsframe in rsframes:
        if rsframe.is_depth_frame():
            # depth_frame = rsframes.get_depth_frame()
            colorizer = rs.colorizer()

            # Create alignment primitive with color as its target stream:
            align = rs.align(rs.stream.color)
            rsframes = align.process(rsframes)

            # Update color and depth frames:
            aligned_depth_frame = rsframes.get_depth_frame()
            colorized_depth = np.asanyarray(
                colorizer.colorize(aligned_depth_frame).get_data())
     
            # UNCOMMENT TO SEE DEPTH FRAME
            cv2.imshow("frame", colorized_depth)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


########### Controlling the Car ############
    ''' 
    There are two functions used to control the car, drive() and steer() 

    drive() takes as input numbers from 0 to about 2. 0 = Stopped, 2 = Fast forward
        Note: The car can't drive backwards
    steer() takes as input numbers from -30 to 30. -30 = far left, 30 = far right
    '''
    
    # Example: (The car should drive forward/turn right for four seconds, drive slower/turn left for four seconds, then straighten and stop)
    
    # UNCOMMENT TO CONTROL CAR
    #drive(.8)
    #steer(16)
    #time.sleep(4)
    #drive(.4)
    #steer(-20)
    #time.sleep(4)
    #drive(0)
    #steer(0)
    #time.sleep(4)
    

vs.release()










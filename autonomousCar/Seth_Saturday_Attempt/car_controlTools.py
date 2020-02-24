"""
Library of the various control tools we are using
for the autonomous car project
"""

import numpy as np
import serial
import time

def decisionGridGaussian(nx, ny, sigx=1., sigy=1., gain=1.):
    """
    Decision grid based on a gaussian distribution function
    """
    x, y = np.meshgrid(np.linspace(-nx/2, nx/2, nx), np.linspace(0, ny, ny))

    dg = np.exp(-((x/(2.*sigx))**2 + (y/(2.*sigy))**2))

    m = np.amax(dg)
    dg = gain*dg/m

    # negate half of the image
    dg[..., int(nx/2):] *= -1

    return dg, x, y

def plt_decisionGrid(dg, x, y, block=True):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt
    plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot_surface(x, y, dg)
    plt.show(block=block)

class carControl:
    def __init__(self):
        # initialize communication with the arduino
        ser = serial.Serial("/dev/ttyUSB0", 115200)
        ser.flushInput()
        time.sleep(2)
        self.ser = ser
        ser.write("!start1580\n".encode())
        ser.write("!speed0.0\n".encode())
        ser.write("!inits0.5\n".encode())
        ser.write("!straight1475\n".encode())
        ser.write("!kp0.001\n".encode())
        ser.write("!kd0.001\n".encode())
        ser.write("!pid0\n".encode())

        self.drive(1.0)
        time.sleep(.1)
        self.drive(0)

    def drive(self, speed):
        forward_command = "!speed" + str(speed) + "\n"
        self.ser.write(forward_command.encode())

    def steer(self, degree):
        steer_command = "!steering" + str(degree) + "\n"
        self.ser.write(steer_command.encode())


if __name__=="__main__":
    cc = carControl()
    cc.drive(1.)
    cc.steer(10)
    time.sleep(3)
    cc.drive(-1)
    cc.steer(-10)

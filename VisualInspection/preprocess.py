#! /usr/bin/env python3

"""
Pre-process image data by segmenting it and then extracting
regions around segmented objects
"""

import argparse

parser = argparse.ArgumentParser(description='Visual inspection preprocessor')


parser.add_argument('-i', '--input_file', help='name of video file to process')
parser.add_argument('-o', '--output_file', help='name of image files to output, for example -o almonds will output almonds_0001.jpg...')

args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt
import cv2


rawVideo = cv2.VideoCapture(args.input_file)

backSub = cv2.createBackgroundSubtractorMOG2()

while (rawVideo.isOpened()):
    ret, frame = rawVideo.read()

    cv2.imshow('Raw Video', frame)
    cv2.waitKey(15)

rawVideo.release()
cv.destroyAllWindows()


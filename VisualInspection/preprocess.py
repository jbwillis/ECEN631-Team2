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

fov_tl = (50, 0) # top left
fov_br = (590, 480) # bottom right

rawVideo = cv2.VideoCapture(args.input_file)

backSub = cv2.createBackgroundSubtractorMOG2()

def segment(img):
    """
    """
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = 1.5*gray.astype(float)
    gray = np.uint8(gray)

    ret, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((5,5),np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    processed, contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # remove contours with points outside crop region

    return processed


while (rawVideo.isOpened()):
    ret, frame = rawVideo.read()

    processed = segment(frame)

    frame = cv2.rectangle(frame, fov_tl, fov_br, (0, 255, 0))

    cv2.imshow('Raw Video', frame)
    cv2.imshow('Processed', processed)
    cv2.waitKey(15)

rawVideo.release()
cv.destroyAllWindows()


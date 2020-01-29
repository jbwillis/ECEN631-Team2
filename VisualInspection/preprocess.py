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

def segment(img):
    """
    """
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    fgMask = backSub.apply(blurred)

    processed = fgMask
    # convert to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.adaptiveThreshold(gray, 127, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1)
    # processed = gray

    # processed, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return processed


while (rawVideo.isOpened()):
    ret, frame = rawVideo.read()

    processed = segment(frame)

    cv2.imshow('Raw Video', frame)
    cv2.imshow('Processed', processed)
    cv2.waitKey(15)

rawVideo.release()
cv.destroyAllWindows()


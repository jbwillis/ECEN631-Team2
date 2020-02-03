from keras.models import load_model
import argparse
import pickle
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to test file")
ap.add_argument("-m", "--model", required=True,
	help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to label binarizer")
args = vars(ap.parse_args())

fov_tl = (50, 0) # top left
fov_br = (610, 480) # bottom right

seg_size = (75, 75) # half of the box size
offsets = np.array([[-75, 75], [-75, 75]])

rawVideo = cv2.VideoCapture(args["video"])

backSub = cv2.createBackgroundSubtractorMOG2()

model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

def processAndLabel(seg):
    seg = cv2.resize(seg, (32, 32)).flatten()
    seg = seg.astype("float")/255.0
    seg = seg.flatten()
    seg = seg.reshape((1, seg.shape[0]))

    preds = model.predict(seg)

    i = preds.argmax(axis=1)[0]

    return lb.classes_[i]



def segment(img):
    """
    """
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = 1.5*gray.astype(float)
    gray = np.uint8(gray)

    ret, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((5,5),np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    # set edges to 0
    opened[0:480, 0:50] = 0
    opened[0:480, 610:] = 0

    processed, contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) > 0):
        maxC = max(contours, key = cv2.contourArea)
        (x, y), rad = cv2.minEnclosingCircle(maxC)
        cent = (int(x), int(y))
        rad = int(rad)
    else:
        cent = None
        rad = None

    # remove contours with points outside crop region

    return processed, cent, rad

segnum = 0
while (rawVideo.isOpened()):

    # only process every third frame
    for i in range(3):
        ret, frame = rawVideo.read()

    processed, cent, rad = segment(frame)


    seg = None
    label = None
    if cent is not None and rad is not None:
        corners = np.array([[cent[0], cent[0]], [cent[1], cent[1]]]) + offsets
        
        if corners[1,0] > 0 and corners[1,1] < 480 and corners[0,0] > 50 and corners[0,1] < 610:
            seg = np.copy(frame[corners[1,0]:corners[1,1], corners[0,0]:corners[0,1]])

            label = processAndLabel(seg)

            cv2.imshow('segment', seg)
            cv2.waitKey(5)
            segnum += 1

        frame = cv2.circle(frame, cent, rad, (0, 0, 255))

     
    frame = cv2.rectangle(frame, fov_tl, fov_br, (0, 255, 0))
    if label is not None:
        frame = cv2.putText(frame, label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Raw Video', frame)
    cv2.imshow('Processed', processed)
    cv2.waitKey(50)

rawVideo.release()
cv2.destroyAllWindows()



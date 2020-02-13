import cv2 as cv

def main():
    vs = cv.VideoCapture("/dev/video2", cv.CAP_V4L)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('transform.avi',fourcc, 20.0, (640,480))
    while True:
        ret, frame = vs.read()
        out.write(frame)
        #cv.imshow("frame", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    out.release()

if __name__=="__main__":
    main()

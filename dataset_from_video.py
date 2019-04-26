import numpy as np
import cv2 as cv


def main():
    cap = cv.VideoCapture('data/source/src.mp4')

    frames = []
    try:
        while True:

            _, frame = cap.read()

            frame = cv.resize(frame, (64, 64))
            frames.append(frame)
            cv.imshow('frame', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    except:
        pass

    np.save('data/hX.npy', np.asarray(frames))


if __name__ == '__main__':
    main()
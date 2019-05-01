import os
import cv2
import numpy as np
from network.model import Gan


def main():
    """
        I know, exists easier method to do this, but  I want to save all generated frames.
    """
    size = (64, 64, 3)
    model = Gan(input_shape=size, image_shape=(64, 64, 6))
    model.build_train_functions()
    model.load_weights()

    # Writes video
    frames = 300

    ones = np.ones((1, 64, 64, 3))
    prediction = model.generator.predict(ones)
    prediction = np.float32(prediction)[:, :, :, 1:4]
    cv2.imwrite('data/gen_frames/frame{i}.jpg'.format(i=0), np.reshape(prediction * 255, [size[0], size[1], size[2]]))
    for i in range(frames):
        prediction = model.generator.predict(np.reshape(prediction, [1, size[0], size[1], size[2]]))
        prediction = np.float32(prediction)[:, :, :, 1:4]
        cv2.imwrite('data/gen_frames/frame{i}.jpg'.format(i=i+1), np.reshape(prediction * 255, [size[0], size[1], size[2]]))

    # Writes video from constructed frames
    _, _, src_files = next(os.walk('data/gen_frames/'))
    file_count = len(src_files)
    img_array = []
    for i in range(file_count):

        img = cv2.imread('data/gen_frames/frame{index}.jpg'.format(index=i))
        if img is None:
            continue
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('data/gen_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    print("Video converted.")


if __name__ == "__main__":
    main()
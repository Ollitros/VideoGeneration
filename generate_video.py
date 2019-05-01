import cv2
import numpy as np
from network.model import Gan


def main():
    size = (64, 64, 3)
    model = Gan(input_shape=size, image_shape=(64, 64, 6))
    model.build_train_functions()
    model.load_weights()

    # Writes video
    frames = 500
    img_array = []

    noise = np.random.normal(0, 1, (1, 64, 64, 3))
    prediction = model.generator.predict(noise)
    prediction = np.float32(prediction)[:, :, :, :3]
    img_array.append(np.reshape(prediction * 255, [size[0], size[1], size[2]]))
    for i in range(frames):
        prediction = model.generator.predict(prediction)
        prediction = np.float32(prediction)[:, :, :, :3]
        img_array.append(np.reshape(prediction * 255, [size[0], size[1], size[2]]))

    out = cv2.VideoWriter('data/generated_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (size[0], size[1]))

    for i in range(len(img_array)):
        out.write(np.uint8(img_array[i]))
    out.release()

    print("Video converted.")


if __name__ == "__main__":
    main()
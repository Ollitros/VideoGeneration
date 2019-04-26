import numpy as np
import cv2 as cv
import time
from network.model import Gan


def train_model(input_shape, train_x, train_y, epochs, batch_size):
    model = Gan(input_shape=input_shape, image_shape=(64, 64, 6))
    model.build_train_functions()

    errG_sum = errD_sum = 0
    display_iters = 1

    t0 = time.time()
    iters = train_x.shape[0] // batch_size

    model.load_weights()
    for i in range(epochs):

        # Train discriminator
        step = 0
        for iter in range(iters):
            fake = model.generator.predict(train_x[step:step + batch_size])
            errD = model.train_discriminator(X=np.float32(fake)[:, :, :, :3], Y=train_y[step:step + batch_size])
            step = step + batch_size
        errD_sum += errD[0]

        # Train generator
        step = 0
        for iter in range(iters):
            errG = model.train_generator(X=train_x[step:step + batch_size], Y=train_y[step:step + batch_size])
            step = step + batch_size
        errG_sum += errG[0]

        # Visualization
        if i % display_iters == 0:

            print("----------")
            print('[iter %d] Loss_D: %f Loss_G: %f  time: %f' % (i, errD_sum / display_iters,
                                                                 errG_sum / display_iters, time.time() - t0))
            print("----------")
            display_iters = display_iters + 1

        if i % 10 == 0:
            # Makes predictions after each epoch and save into temp folder.
            prediction = model.generator.predict(train_x[0:2])
            cv.imwrite('data/models/temp/image{epoch}.jpg'.format(epoch=i + 0), prediction[0] * 255)

    model.save_weights()


def main():

    # Load the dataset
    X = np.load('data/source/X.npy')
    X = X.astype('float32')
    X /= 255

    epochs = 100
    batch_size = 5
    input_shape = (64, 64, 3)

    train_x = []
    for i in range(100):
        train_x.append(X[0])
    train_y = []
    for i in range(100):
        train_y.append(X[10])

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    train_model(input_shape=input_shape, train_x=train_x, train_y=train_y, epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    main()


# X = np.load("data/rawX.npy")
# print(X.shape)
# array = []
# for i in range(X.shape[0]):
#     temp = cv.resize(X[i], (64, 64))
#     array.append(temp)
# np.save('data/X.npy', np.asarray(array))
#
# X = np.load('data/X.npy')
# print(X.shape)
# for i in range(X.shape[0]):
#     cv.imshow('frame', X[i])
#     cv.waitKey(0)
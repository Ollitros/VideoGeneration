import numpy as np
import cv2 as cv
import time
from network.model import Gan


def train_model(input_shape, train_x, epochs, batch_size):
    """
        In standard GANs discriminator take input from generator and from real data, then tries to classify them.
        This implementation works different - discriminator take both inputs as real data, but the main process
        is going in adversarial loss which overcompensate standard approach. Such implementation verified experimentally.

    """
    model = Gan(input_shape=input_shape, image_shape=(64, 64, 6))
    model.build_train_functions()

    errG_sum = errD_sum = 0
    display_iters = 1

    t0 = time.time()
    model.load_weights()

    iters = np.asarray(train_x).shape[0] // batch_size
    if np.asarray(train_x).shape[0] - iters * batch_size == 0:
        train_x = train_x.tolist()
        train_x.append(train_x[-1])
    train_x = np.asarray(train_x)
    for epoch in range(epochs):

        # Train generator predict first frame form noise
        # Train discriminator
        batch_y = []
        for f in range(batch_size):
            batch_y.append(train_x[0])

        batch_noise = np.random.normal(0, 1, (batch_size, input_shape[0], input_shape[1], input_shape[2]))
        errD = model.train_discriminator(X=batch_noise, Y=batch_y)
        errD_sum += errD[0]
        # Train generator
        errG = model.train_generator(X=batch_noise, Y=batch_y)
        errG_sum += errG[0]

        # Train others frames
        # Train discriminator
        step = 0
        for iter in range(iters):
            errD = model.train_discriminator(X=train_x[step: (step + batch_size)], Y=train_x[(step + 1): (step + 1 + batch_size)])
            step = step + batch_size
        errD_sum += errD[0]

        # Train generator
        step = 0
        for iter in range(iters):
            errG = model.train_generator(X=train_x[step:(step + batch_size)], Y=train_x[(step + 1):(step + 1 + batch_size)])
            step = step + batch_size
        errG_sum += errG[0]

        # Visualization
        if epoch % display_iters == 0:
            print("----------")
            print('[iter %d] Loss_D: %f Loss_G: %f  time: %f' % (epoch, errD_sum / display_iters,
                                                                 errG_sum / display_iters, time.time() - t0))
            print("----------")
            display_iters = display_iters + 1

        if epoch % 10 == 0:
            # Makes predictions after each epoch and save into temp folder.
            prediction = model.generator.predict(train_x[0:2])
            prediction = np.float32(prediction[0] * 255)[:, :, 1:4]
            cv.imwrite('data/models/temp/image{epoch}.jpg'.format(epoch=epoch + 0), prediction)
            model.save_weights()

    model.save_weights()


def main():

    # Load the dataset
    X = np.load('data/source/X.npy')
    X = X.astype('float32')
    X /= 255

    epochs = 100
    batch_size = 5
    input_shape = (64, 64, 3)
    train_model(input_shape=input_shape, train_x=X, epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    main()



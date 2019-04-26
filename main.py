import numpy as np
import cv2 as cv
import time
from network.model import Gan


def train_model(input_shape, train_x, epochs, batch_size):
    model = Gan(input_shape=input_shape, image_shape=(64, 64, 6))
    model.build_train_functions()

    errG_sum = errD_sum = 0
    display_iters = 1

    t0 = time.time()
    model.load_weights()
    for epoch in range(epochs):
        for i in range(train_x.shape[0]):
            if i == 0:

                # Train discriminator
                batch_y = []
                for f in range(batch_size):
                    batch_y.append(train_x[f])

                batch_noise = np.random.normal(0, 1, (batch_size, input_shape[0], input_shape[1], input_shape[2]))
                fake = model.generator.predict(batch_noise)
                errD = model.train_discriminator(X=np.float32(fake)[:, :, :, :3], Y=batch_y)
                errD_sum += errD[0]

                # Train generator
                errG = model.train_generator(X=batch_noise, Y=batch_y)
                errG_sum += errG[0]
            elif i == (train_x.shape[0] - 1):
                break
            else:
                # Train discriminator
                batch_x = []
                for k in range(batch_size):
                    batch_x.append(train_x[k])
                batch_y = []
                for v in range(batch_size):
                    batch_y.append(train_x[v + 1])
                batch_x = np.asarray(batch_x)
                batch_y = np.asarray(batch_y)
                fake = model.generator.predict(np.asarray(batch_x))
                errD = model.train_discriminator(X=np.float32(fake)[:, :, :, :3], Y=batch_y)
                errD_sum += errD[0]

                # Train generator
                errG = model.train_generator(X=batch_x, Y=batch_y)
                errG_sum += errG[0]

        # Visualization
        if epoch % display_iters == 0:
            print("----------")
            print('[iter %d] Loss_D: %f Loss_G: %f  time: %f' % (epoch, errD_sum / display_iters,
                                                                 errG_sum / display_iters, time.time() - t0))
            print("----------")
            display_iters = display_iters + 1

        if epoch % 1 == 0:
            # Makes predictions after each epoch and save into temp folder.
            prediction = model.generator.predict(train_x[0:2])
            cv.imwrite('data/models/temp/image{epoch}.jpg'.format(epoch=epoch + 400), prediction[0] * 255)

    model.save_weights()


def main():

    # Load the dataset
    X = np.load('data/source/X.npy')
    X = X.astype('float32')
    X /= 255

    epochs = 100
    batch_size = 5
    input_shape = (64, 64, 3)

    train_x = np.asarray(X)
    train_model(input_shape=input_shape, train_x=train_x, epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    main()



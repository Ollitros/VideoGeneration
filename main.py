import numpy as np
import cv2 as cv
from keras.datasets import mnist
from model import GAN

# Load the dataset
X = np.load('data/X.npy')
X = X.astype('float32')
X /= 255
input_shape = X.shape

model = GAN(input_shape=input_shape, latent_dim=100)
model.train(X, epochs=10000, batch_size=20)

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
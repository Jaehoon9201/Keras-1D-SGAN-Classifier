
from numpy import expand_dims
from keras.models import load_model
from keras.datasets.mnist import load_data
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os
from keras.utils import to_categorical
from numpy.random import randn
from keras.models import load_model
from matplotlib import pyplot
from math import sqrt
from numpy import asarray

df2 = pd.read_csv('train_data.csv')
err_train = df2.values[:, :-1]
status_train = df2.values[:, -1].astype(np.int64)
status_train_cat = to_categorical(status_train)

df_test2 = pd.read_csv('test_data.csv')
err_test = df_test2.values[:, :-1]
status_test = df_test2.values[:, -1].astype(np.int64)





def generate_latent_points(latent_dim, n_samples, n_class):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = asarray([n_class for _ in range(n_samples)])
    return [z_input, labels]

colors = ['r', 'g','c','b', 'm','k', 'y','r', 'g','c','b' ]


# load the model
model = load_model('model/c_model_12960.h5')
# load the dataset
(trainX, trainy) = (err_train, status_train)
(testX, testy) = (err_test, status_test)
# expand to 3d, e.g. add channels
trainX = expand_dims(trainX, axis=-1)
testX = expand_dims(testX, axis=-1)

_, train_acc = model.evaluate(trainX, trainy, verbose=0)
print('Train Accuracy: %.3f%%' % (train_acc * 100))
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Test Accuracy: %.3f%%' % (test_acc * 100))


latent_dim = 12
n_examples = 30000
n_class = 7
# generate images
latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
# generate images
X = model.predict([latent_points, labels])
# plot the result
#save_plot(X, n_examples)


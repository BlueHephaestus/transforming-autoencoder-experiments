import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

import numpy as np

X_train = np.concatenate((np.random.rand(500, 50, 50)*(0.5-1e-7), np.random.rand(500, 50, 50)*(0.5-1e-7)+(0.5+1e-7)), axis=0)
X_train = np.reshape(X_train, [-1, 50, 50, 1])
Y_train = np_utils.to_categorical(np.concatenate((np.zeros(shape=(500), dtype=int), np.ones(shape=(500), dtype=int)), axis=0), 2)

X_test = np.concatenate((np.random.rand(50, 50, 50)*0.499, np.random.rand(50, 50, 50)*0.499+0.501), axis=0)
X_test = np.reshape(X_test, [-1, 50, 50, 1])
Y_test = np_utils.to_categorical(np.concatenate((np.zeros(shape=(50), dtype=int), np.ones(shape=(50), dtype=int)), axis=0), 2)

#Build model
#d = 1.0-1e-7
d = 0.0
model = Sequential()
model.add(Convolution2D(32, 5, 5, border_mode='same', input_shape=(50, 50, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
#model.add(Dropout(d))
model.add(Convolution2D(32, 5, 5))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(d))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation("relu"))

model.add(Dropout(d))

model.add(Dense(2))
model.add(Activation("softmax"))

#We are concluding that having dropout at the end is equivalent to having it be the same percentage at each point

"""
model.add(Dense(input_dim=50, output_dim=64))
model.add(Activation("sigmoid"))
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))
"""

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

results = model.fit(X_train, Y_train, nb_epoch=4, batch_size=10)

print results.history["loss"]
print results.history["acc"]

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

print loss_and_metrics

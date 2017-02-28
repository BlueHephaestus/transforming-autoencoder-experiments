"""
Minimal MNIST NN example

-Blake Edwards / Dark Element
"""
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

"""
Convert to float32
"""
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

"""
Feature Scale
"""
X_train /= 255
X_test /= 255

"""
Create One-hot label matrices
"""
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

model = Sequential()
#model.add(Flatten(input_shape=(28,28)))
model.add(Reshape((28*28,), input_shape=(28,28) ))
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
results = model.fit(X_train, Y_train, nb_epoch=4, batch_size=10)

print results.history["loss"]
print results.history["acc"]

loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

print loss_and_metrics

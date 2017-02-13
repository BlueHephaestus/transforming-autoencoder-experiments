"""
MNIST Autoencoder experiments

-Blake Edwards / Dark Element
"""
"""
OI FUTURE SELF

This currently gets about 28% accuracy with the current parameters
    See how we can get better accuracy with different parameters, and how to get to what is a normal accuracy (if this isn't normal, which it might be)
        You should read up on how to parameterize autoencoders, often used setups, etc.
        got params from https://blog.keras.io/building-autoencoders-in-keras.html
    It is learning though, which is what is important. We have here a working, small autoencoder for MNIST with Keras.
    Fuck you torch, this shit took like 30 min.
    Then print the results to see how good they look, and we should be good here.
    That's it, as far as I can think right now! Good luck, have fun!
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

"""
Create Model
"""
model = Sequential()
#model.add(Flatten(input_shape=(28,28)))
model.add(Reshape((28*28,), input_shape=(28,28) ))
model.add(Dense(32))
model.add(Activation("relu"))
model.add(Dense(28*28))
model.add(Activation("sigmoid"))
model.add(Reshape((28,28,)))

"""
Compile and Train Model
"""
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
results = model.fit(X_train, X_train, nb_epoch=50, batch_size=256, shuffle=True)

print results.history["loss"]
print results.history["acc"]

"""
Test Model
"""
loss_and_metrics = model.evaluate(X_test, X_test, batch_size=256)

print loss_and_metrics

"""
MNIST Autoencoder experiments

-Blake Edwards / Dark Element
"""
"""
This currently gets about 28% accuracy with the current parameters, and I learned that accuracy is completely irrelevant for autoencoders.
This gets really good recreations of MNIST images, is what is important.
"""
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import cv2
import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

def disp_img_fullscreen(img, name="test"):
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dense(28*28))
model.add(Activation("sigmoid"))
model.add(Reshape((28,28,)))

"""
Compile and Train Model
"""
model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
results = model.fit(X_train, X_train, nb_epoch=8, batch_size=50, shuffle=True)

#print results.history["loss"]
#print results.history["acc"]

"""
Test Model
"""
loss_and_metrics = model.evaluate(X_test, X_test, batch_size=256)

#print loss_and_metrics

"""
Display some outputs:
    Get a small portion of the test set, and get reconstructed samples
"""
display_test_n = 16
actual_samples = X_test[:display_test_n]
reconstructed_samples = model.predict(actual_samples, batch_size=1, verbose=0)

"""
Then create a row of actual samples, and a row of reconstructed samples.
We can then stack them on top of each other for easy multiple image comparison.
"""
actual_samples = np.concatenate([actual_sample for actual_sample in actual_samples], axis=1)
reconstructed_samples = np.concatenate([reconstructed_sample for reconstructed_sample in reconstructed_samples], axis=1)
disp_img_fullscreen(np.concatenate((actual_samples, reconstructed_samples), axis=0))

"""
Some simple tests with translating / rotating / dilating / etc. MNIST images.
"""
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import cv2
import numpy as np

img_h = 28
img_w = 28

def disp_img_fullscreen(img, name="test"):
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_concatenated_row(samples):
    """
    Concatenate each sample in samples horizontally, along axis 1.
    Return the resulting array.
    """
    return np.concatenate([sample for sample in samples], axis=1)

def get_concatenated_col(samples):
    """
    Concatenate each sample in samples vertically, along axis 0.
    Return the resulting array.
    """
    return np.concatenate([sample for sample in samples], axis=0)

def transform_samples(samples, T):
    """
    Given a 3d matrix samples and a transformation matrix T,
    return a new 3d samples matrix with each image transformed appropriately
    """
    transformed_samples = np.zeros_like(samples)
    img_h, img_w = samples.shape[-2:]
    for i, sample in enumerate(samples):
        transformed_samples[i] = cv2.warpAffine(sample, T, (img_w, img_h))
    return transformed_samples

def translate_samples(samples, dx, dy):
    """
    Given a 3d matrix samples, the translation delta x and delta y,
    return a new 3d samples matrix with each image translated appropriately.
    """
    """
    First generate translation matrix for cv2's warpAffine function
    """
    T = np.float32([[1,0,dx], [0,1,dy]])

    return transform_samples(samples, T)

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
Display some outputs:
    Get a small portion of the test set, and get transformed samples
"""
display_test_n = 16
actual_samples = X_test[:display_test_n]
x_translated_samples = translate_samples(actual_samples, 0, 10)
y_translated_samples = translate_samples(actual_samples, 10, 0)

"""
Then create a row of actual samples, and a row for each transformation's samples.
We can then stack them on top of each other for easy multiple image comparison.
"""
actual_samples = get_concatenated_row(actual_samples)
x_translated_samples = get_concatenated_row(x_translated_samples)
y_translated_samples = get_concatenated_row(y_translated_samples)
comparison_img = get_concatenated_col((actual_samples, x_translated_samples, y_translated_samples))
disp_img_fullscreen(comparison_img)

import cv2
import numpy as np
h = 20.
w = 10.
a,b = np.meshgrid(np.arange(h),np.arange(w))
a /= h
b /= w
print a
print b

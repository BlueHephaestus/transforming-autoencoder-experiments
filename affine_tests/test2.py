import cv2
import numpy as np
a = np.random.randn(28,28)
#cv2.imshow('asdf',a)
#cv2.waitKey(0)
a *= 255.0
cv2.imwrite('0.jpg', a)

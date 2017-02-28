"""
Some simple tests with translating / rotating / dilating / etc. MNIST images,
    using our technique for decoding multiple Atomic Capsule Renders
    so that they can be added together to get the combined output of multiple ACs 
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

def create_transformation_matrix(theta, z, sx, sy, dx, dy, h, w):
    """
    Given human readable parameters for translation (dx & dy), scaling (sx & sy), shearing (z), and rotation (theta), as well as the image height and width (h & w),
        We move the center of the image to the origin for more intuitive transformations using normalization translations,
        Then apply them in the order rotation -> shearing -> scaling -> translation

    Would we be better suited adding in shearing in the y direction?
    """

    """
    Set our normalization matrices for moving the origin of transformation to the center of the image.
    Moves top left corner of frame to center of frame, by using dx = w/2 and dy = h/2
        [1         , 0         ,.5*w ]
        [0         , 1         ,.5*h ]
        [0         , 0         ,  1  ]
    Moves center of frame to top left corner of frame, the inverse of our previous transformation, by using dx = -w/2 and dy = -h/2
        [1         , 0         ,-.5*w]
        [0         , 1         ,-.5*h]
        [0         , 0         ,  1  ]
    """
    top_left_to_center = np.float32([[1, 0, .5*w], [0, 1, .5*h], [0,0,1]])
    center_to_top_left = np.float32([[1, 0, -.5*w], [0, 1, -.5*h], [0,0,1]])

    """
    Create our transformation matrices

    Rotation (no rotation: theta = 0):
        *Note: theta given in degrees, converted to radians
        [cos(theta), -sin(theta), 0]
        [sin(theta), cos(theta),  0]
        [0         , 0         ,  1]

    Shearing (no shearing: z = 0):
        [1         , z         ,  0]
        [0         , 1         ,  0]
        [0         , 0         ,  1]

    Scaling (no scaling: sx = sx = 1):
        [sx        , 0         ,  0]
        [0         , sy        ,  0]
        [0         , 0         ,  1]

    Translation (no translating: dx = dy = 0):
        [1         , 0         , dx]
        [0         , 1         , dy]
        [0         , 0         ,  1]

    """
    theta = theta * np.pi / 180.#Degrees -> Radians
    rotation = np.float32([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0,0,1]])
    shearing = np.float32([[1,z,0],[0,1,0],[0,0,1]])
    scaling = np.float32([[sx,0,0],[0,sy,0],[0,0,1]])
    translation = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])

    """
    Create our final transformation matrix by chaining these together in a dot product of the form:
        res = top_left_to_center * translation * scaling * shearing * rotation * center_to_top_left.
    This is because if we want to apply them in our order, we would normally get the inverse of the dot product
        (translation * scaling * shearing * rotation)^-1 ,
    But since inverses are computationally expensive we instead just flip the order, as that is equivalent in this scenario.
    We do not flip the order of the normalization transformations, however.
    """
    transformation = top_left_to_center.dot(translation).dot(scaling).dot(shearing).dot(rotation).dot(center_to_top_left)
    return transformation

def transform_samples(samples, T, sub_h, sub_w, img_h, img_w):
    """
    Given a vector of matrix image samples:
        1. Generate a random transformation matrix
        2. For each sample:
            a. Create zeroed copy of sample
            b. Divide into 4 parts, 4x4x28x28
            c. Apply our randomly generated transformation matrix to each part
            d. Sum parts onto zeroed copy
        3. Return transformed samples
    """
    #clear_r3 = np.array([[1,0,0],[0,1,0],[0,0,0]])
    #T = np.eye(3) + 0.05*(np.dot(clear_r3,np.random.randn(3,3)))

    transformed_samples = np.zeros_like(samples)
    for sample_i, sample in enumerate(samples):
        """
        we don't get subsections of our images the conventional way when we do this with our network,
            we get them with our weights. We also end up with an image the same size as our original image
        So, when testing, we can do it however we want.
        """
        parts = np.zeros((img_h//sub_h, img_w//sub_w, img_h, img_w))
        for row_i in range(0, len(sample), sub_h):
            for col_i in range(0, len(sample), sub_w):
                """
                Get relative indices
                """
                parts_row_i = row_i//sub_h
                parts_col_i = col_i//sub_w

                """
                Get our subsection, and place it in the corresponding location in our result array.
                """
                sub = sample[row_i:row_i+sub_h, col_i:col_i+sub_w]
                parts[parts_row_i, parts_col_i, row_i:row_i+sub_h, col_i:col_i+sub_w] = sub
                #disp_img_fullscreen(parts[parts_row_i, parts_col_i])
        """
        Now we apply our transformation matrix to each part
        OI FUTURE SELF
            Write some code that does this for any nxn matrix given

        """
        for parts_row in parts:
            for part in parts_row:
                """
                Apply affine transformation
                """
                trans_part = cv2.warpAffine(part, T[:2], (img_w,img_h))
                
                """
                Add to correct location in transformed_samples result
                """
                transformed_samples[sample_i] += trans_part
    return transformed_samples

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
sub_h = 7
sub_w = 7
img_h = 28
img_w = 28

display_test_n = 16
actual_samples = X_test[:display_test_n]
for actual_sample_i, actual_sample in enumerate(actual_samples):
    cv2.imwrite("%s.jpg"%(str(actual_sample_i)), actual_sample*255.0)
sys.exit()

T = create_transformation_matrix(45, 0.0, 1.0, 1.0, 0, 0, img_h, img_w)
#T = np.eye(3)+0.1*np.random.randn(3,3)
transformed_samples = transform_samples(actual_samples, T, sub_h, sub_w, img_h, img_w)
actual_samples = get_concatenated_row(actual_samples)
transformed_samples = get_concatenated_row(transformed_samples)
comparison_img = get_concatenated_col((actual_samples, transformed_samples))
disp_img_fullscreen(comparison_img)

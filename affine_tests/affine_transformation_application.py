import numpy as np
import cv2

def disp_img_fullscreen(img, name="test"):
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("2d_grid.png")
h, w, c = img.shape

theta = 90
theta = -theta * np.pi / 180.#Degrees -> Radians
dx = 200
dy = -200

"""
Set our normalization matrices for moving the origin of transformation to the center of the image.
"""
top_left_to_center = np.float32([[1, 0, .5*w], [0, 1, .5*h], [0,0,1]])#Moves top left corner of frame to center of frame
#center_to_top_left = np.linalg.inv(top_left_to_center)
center_to_top_left = np.float32([[1, 0, -.5*w], [0, 1, -.5*h], [0,0,1]])#Moves center of frame to top left corner of frame, the inverse of our previous transformation

def create_transformation_matrix(theta, z, sx, sy, dx, dy):
    """
    Given human readable parameters for translation (dx & dy), scaling (sx & sy), shearing (z), and rotation (theta),
        We move the center of the image to the origin for more intuitive transformations,
        Then apply them in the order rotation -> shearing -> scaling -> translation

    Would we be better suited adding in shearing in the y direction?
    """

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

"""
this does it in the opposite order of multiplication for the transformations we care about (not the normalization ones)
so if we want to rotate, then translate, we would do 
    translation * rotation
and if we want to translate, then rotate, we would do
    rotation * translation
It is the inverse of the matrix obtained doing it in the human readable way.
But since we don't want to do the inverse, we do it our own way, by re-ordering the operations
"""
#T = top_left_to_center.dot(rotate_theta).dot(translation).dot(center_to_top_left) 
#T = top_left_to_center.dot(translation).dot(rotate_theta).dot(center_to_top_left)
#def create_transformation_matrix(theta, z, sx, sy, dx, dy):
#T = create_transformation_matrix(90, 0, 1, 1, 200, 200)
T = create_transformation_matrix(45, 0.0, 1.0, 1.0, 200, 200)
trans_img = cv2.warpAffine(img, T[:2], (w,h))
disp_img_fullscreen(trans_img)
"""
n = 10
sigma = 0.1
for i in range(n):
    T = np.float32([[1,0,0],[0,1,0]]) + sigma*np.random.randn(2,3)
    trans_img = cv2.warpAffine(img, T[:2], (w,h))
    disp_img_fullscreen(trans_img)
"""

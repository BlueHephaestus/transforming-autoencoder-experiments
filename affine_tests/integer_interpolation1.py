"""
Same as mk1 interpolation, except uses full integer interpolation instead of our barebones initial version.
Does not always give us the results we want, is lossy when doing large scales in preliminary tests.
We will likely use bilinear interpolation over this, as it should scale and do lossy transforms
    while still retaining a nice representation of the image
This is just a proof of concept for the integer interpolation algorithm, 
    and I will likely not use this anymore but continue work on the project using bilinear interpolation over this.
Example:
    Scaling by > 1.5: massive holes in result
    Using random generation of T method with sigma = 0.1: multiple holes in result
"""
import numpy as np
import cv2

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

"""Load image"""
src_img = cv2.imread("0.jpg")

"""Get Greyscale, since we are only testing with one channel atm"""
#src_img = src_img[0:28,0:28,0]
src_img = src_img[:, :, 0]
#src_img = np.random.randn(28,28)
#print src_img.shape
#disp_img_fullscreen(src_img)

"""Create affine transformation matrix"""
#T = np.array([[1,0,-5],[0,1,10],[0,0,1]])
#T = np.float32([[1.2,0,0],[0,1.2,0],[0,0,1]])
clear_r3 = np.array([[1,0,0],[0,1,0],[0,0,0]])
T = np.eye(3) + 0.10*(np.dot(clear_r3,np.random.randn(3,3)))

"""Initialize result image and get image dims"""
dst_img = np.zeros_like(src_img)
h, w = src_img.shape

"""Create src meshgrid"""
x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

"""Add extra dimension so we can concatenate together to have a vector of cords for each one"""
x = x[:,:, None]
y = y[:,:, None]

"""
Concatenate as (y,x) instead of (x, y) so that we can reference as (h,w) or (row, col) instead of the other way around.
    So yea, annoying tensor indexing is to blame.
"""
src_meshgrid = np.concatenate([y,x, np.ones_like(x)], axis=2)
src_meshgrid = src_meshgrid[:,:,:,None]

#Debugging
#print src_meshgrid.shape
#print src_meshgrid[0][0], src_meshgrid[-1][0], src_meshgrid[0][-1], src_meshgrid[-1][-1]

"""
Apply our transformation matrix to src meshgrid, and put the result in dst meshgrid.
Since we seem to need to have the last two dimensions of src_meshgrid to be compatible with
    the dimensions of our transformation matrix, 
    so that our dot product essentially becomes 3x1 * 3x3,
    We have to add the empty dimension to our src_meshgrid, 
    And we have to then reshape the result to remove the empty dimension.
"""
dst_meshgrid = np.tensordot(src_meshgrid, T, axes=([2],[1])) 
dst_meshgrid = np.reshape(dst_meshgrid, (h, w, 3))

"""
Since we no longer need the empty dimension, 
    and our following formula is easier with x and y split again,
    We do both in one move
"""
dst_y, dst_x, dst_extra = np.split(dst_meshgrid, 3, 2)
dst_y, dst_x = np.reshape(dst_y, (h,w)), np.reshape(dst_x, (h,w))

#Debugging
#print dst_x.shape, dst_y.shape
#print dst_x[0][0], dst_x[-1][0], dst_x[0][-1], dst_x[-1][-1]
#print dst_y[0][0], dst_y[-1][0], dst_y[0][-1], dst_y[-1][-1]

#Debugging
#print dst_meshgrid.shape
#print dst_meshgrid[0][0], dst_meshgrid[-1][0], dst_meshgrid[0][-1], dst_meshgrid[-1][-1]

"""
Time for our complicated formula.
    Loop through the pixels of our result image, by row then column, i then j.
"""
for i in range(h):
    for j in range(w):
        new_pix = 0
        for n in range(h):
            for m in range(w):
                new_pix += src_img[n,m] * np.logical_not(np.logical_or(np.floor(dst_x[n,m] + .5) - i, np.floor(dst_y[n,m] + .5) - j))
        dst_img[i,j] = new_pix
        
"""
Handle comparison display.
Generate white divider, and concatenate everything for simultaneous comparison.
"""
white_divider = np.ones((src_img.shape[0], 1), dtype=np.uint8)*255
comparison_img = get_concatenated_row((src_img, white_divider, dst_img))
disp_img_fullscreen(comparison_img)

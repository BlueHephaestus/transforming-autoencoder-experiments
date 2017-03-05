"""
-Blake Edwards / Dark Element
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
src_img = cv2.imread("sample4.jpg")
#src_img = cv2.imread("0.jpg")

"""Get Greyscale, since we are only testing with one channel atm"""
#src_img = src_img[0:28,0:28,0]
src_img = src_img[:, :, 0]
#src_img = np.random.randn(192,108)
#print src_img.shape
#disp_img_fullscreen(src_img)

"""Create affine transformation matrix"""
T = np.array([[1,0,5],[0,1,20],[0,0,1]])
#T = np.float32([[1.0,.4,10],[0,1.0,-50],[0,0,1]])
#clear_r3 = np.array([[1,0,0],[0,1,0],[0,0,0]])
#T = np.eye(3) + 0.05*(np.dot(clear_r3,np.random.randn(3,3)))

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

"""
We then get x and y as vectors, since we can more compactly use our formulas this way
"""
#dst_x = dst_x[:,0]
#dst_y = dst_y[0,:]
#dst_y, dst_x = np.reshape(dst_y, (1,w)), np.reshape(dst_x, (h,1))


src_meshgrid = np.reshape(src_meshgrid, (h, w, 3))
src_y, src_x, src_extra = np.split(src_meshgrid, 3, 2)
#src_x = src_x[:,0]
#src_y = src_y[0,:]
#src_y, src_x = np.reshape(src_y, (1,w)), np.reshape(src_x, (h,1))
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
#dst_x /= h
#dst_y /= w
#print np.floor(dst_x)
#print np.floor(dst_y)
"""
dst_x = dst_x[:,0]
dst_y = dst_y[0,:]
dst_x = np.reshape(dst_x, (h,1))
dst_y = np.reshape(dst_y, (1,w))
"""

#for i in range(h):
    #for j in range(w):
a = np.maximum(0, 1-np.abs(dst_x - src_x))
b = np.maximum(0, 1-np.abs(dst_y - src_y))
print n:.all(a==0)
print b
print np.dot(a,b)
sys.exit()
#dst_img[i,j] = np.sum(src_img * np.dot(np.maximum(0, 1-np.abs(dst_x - i)), np.maximum(0, 1-np.abs(dst_y - j))))
#dst_img[i,j] = b.dot(src_img.transpose()).dot(a)
#dst_img = a.dot(src_img).dot(b)
dst_img = src_img * np.dot(a,b)
print dst_img
sys.exit()
"""
Handle comparison display.
Generate white divider, and concatenate everything for simultaneous comparison.
"""
white_divider = np.ones((src_img.shape[0], 1), dtype=np.uint8)*255
comparison_img = get_concatenated_row((src_img, white_divider, dst_img))
disp_img_fullscreen(comparison_img)

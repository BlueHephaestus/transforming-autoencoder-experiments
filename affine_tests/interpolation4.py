"""
MK4 - Realized that there is no feasible way to do this so that it can handle a BxHxWxC tensor all at once, with linear algebra.
I also found that the implementation here https://github.com/qassemoquab/stnbhwd/blob/master/generic/BilinearSamplerBHWD.c 
    used an iterative approach.
It also would have a really high space complexity.
Because of this, I have decided to go with an iterative approach, where we loop through each batch, height, width, and channel in the destination image.
Fortunately, I do still have my optimized equation for this, which means for any given pixel in the output image we don't need to loop at all.

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
#src_img = np.ones_like(src_img)
#src_img = np.random.randn(192,108)
#print src_img.shape
#disp_img_fullscreen(src_img)

"""Initialize result image and get image dims"""
dst_img = np.zeros_like(src_img)
h, w = src_img.shape
"""
Scale src and dst images' pixel values to be 0-1 instead of 0-255,
    since this has shown improved performance over 0-255.
I believe this is because our bilinear interpolation equations apply a weight of 0-1 to 
    each pixel in the src image when getting a pixel value in the dst image, so when we have 0-1
    the formats are the same and it works better.
Not certain of this though.
"""
src_img = src_img.astype(np.float32)
dst_img = dst_img.astype(np.float32)
src_img = src_img/255.
dst_img = dst_img/255.

"""Create affine transformation matrix"""
#T = np.array([[0,0,0],[0,0,0],[0,0,0]])
#T = np.array([[1,0,0],[0,1,0],[0,0,1]])
#T = np.array([[1,0,5],[0,1,20],[0,0,1]])
#T = np.float32([[1.0,.4,10],[0,1.0,-50],[0,0,1]])
#T = np.float32([[1.5,0,0],[0,1.5,0],[0,0,1]])
clear_r3 = np.array([[1,0,0],[0,1,0],[0,0,0]])
T = np.eye(3) + 0.15*(np.dot(clear_r3,np.random.randn(3,3)))


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
dst_x = dst_x[:,0]
dst_y = dst_y[0,:]
dst_y, dst_x = np.reshape(dst_y, (1,w)), np.reshape(dst_x, (h,1))

#Debugging
#print dst_x.shape, dst_y.shape
dst_x, dst_y = dst_x.astype(np.float32), dst_y.astype(np.float32)
#dst_x /= h
#dst_y /= w
#print dst_x[0][0], dst_x[-1][0], dst_x[0][-1], dst_x[-1][-1]
#print dst_y[0][0], dst_y[-1][0], dst_y[0][-1], dst_y[-1][-1]
#sys.exit()

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

"""
for i in range(h):
    for j in range(w):
        c = i / float(h)
        d = j / float(w)
        print c, d
"""
"""
"""
src_img = src_img.astype(np.float32)
dst_img = dst_img.astype(np.float32)
src_img = src_img/255.
dst_img = dst_img/255.
for i in range(h):
    for j in range(w):
        #c = i / float(h)
        #d = j / float(w)
        a = np.maximum(0, 1-np.abs(dst_x - i)).transpose()
        b = np.maximum(0, 1-np.abs(dst_y - j)).transpose()
        #dst_img[i,j] = np.sum(src_img * np.dot(np.maximum(0, 1-np.abs(dst_x - i)), np.maximum(0, 1-np.abs(dst_y - j))))
        #dst_img[i,j] = b.dot(src_img.transpose()).dot(a)
        #print a.dot(src_img).dot(b)

        dst_img[i,j] = a.dot(src_img).dot(b)
        #dst_img[i,j] = np.sum(src_img * np.dot(a,b))
        
#print dst_img
#print np.all(dst_img == src_img)
#sys.exit()
"""
Handle comparison display.
Generate white divider, and concatenate everything for simultaneous comparison.
"""
white_divider = np.ones((src_img.shape[0], 1), dtype=np.uint8)*255
comparison_img = get_concatenated_row((src_img, white_divider, dst_img))
disp_img_fullscreen(comparison_img)

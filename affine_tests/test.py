import numpy as np
import tensorflow as tf
import cv2
import sys

def disp_img_fullscreen(img, name="test"):
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

T = np.float32([[1,0,5],[0,1,5],[0,0,1]])

batch_size = 1
h = 10
w = 5
#flip them so we can combine them and transpose to get a hxwx2
#We don't need t we just need the right distribution of stuffs
data = cv2.imread("2d_grid.png")
h, w, c = data.shape
#data = np.random.randn(batch_size, h, w, 3)#3 for rgb
b, x, y = np.meshgrid(np.arange(batch_size), np.arange(h), np.arange(w), indexing='ij')

#Add extra dimension so we can concatenate together to have a vector of cords for each one
#b = b[:,:,:, None]
x = x[:,:,:, None]
y = y[:,:,:, None]

#print b.shape, x.shape, y.shape
#We don't want the batch index stored at the b, x, y location in our resulting sampling grid, 
#since we aren't doing batch transformations, just 2d x and y coordinate transforms.
#so, we are only concatenating those two. But we also need an entry of 1 for each one to have homogenous coordinates,
#so we do the following:
G = np.concatenate([x,y, np.ones_like(x)], axis=3)
G = G[:,:,:,:,None]
#T = np.reshape(T, (1,1,1,3,3))
print T.shape, G.shape
"""
print data[0, :, 0].shape
print G[0,:,0]
print G[0,:,0].shape
"""

#print G_t[0,:,0]
print G[0][0][1]
"""
G = bxhxwx3x1 want to remove axis 3
T = 3x3 want to remove axis 1
G_t = bxhxwx1x3
G_t -> reshape -> bxhxwx3
G_t -> reshape -> bxhxwx2
"""

G_t = np.tensordot(G, T, axes=([3],[1])) 
G_t = np.reshape(G_t, (batch_size, h, w, 3))
print G_t.shape
G_t = G_t[:,:,:,0:2]
print G_t.shape

"""
Now G_t is our bxhxwx2 sampling grid to match with our bxhxwx3 value array
OI FUTURE SELF
the problem is probably with our take method here
"""
G_t = G_t.astype(np.int64)
data = np.reshape(data, (-1, 3))
G_t = np.reshape(G_t, (-1,2))
print data.shape, G_t.shape
print G_t

data = data.astype(np.int64)
#print type(data[0][0])
#print type(G_t[0][0])
x = tf.placeholder(tf.int64, shape=[None, 3])
i = tf.placeholder(tf.int64, shape=[None, 2])
y = tf.gather_nd(x, i)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
G_t = sess.run(y, feed_dict={x: data, i: G_t})
#G_t = G_t[:,0]*5 + G_t[:, 1]
#G_t = G_t.dot(1 << np.arange(G_t.shape[-1]-1, -1, -1))
#print G_t
#r = np.take(data, G_t, axis=0, mode='clip')
print r.shape
r = np.reshape(r, (batch_size, h, w, 3))
print r.shape
print type(r[0][0][0][0])
disp_img_fullscreen(r[0,:,:,:])
sys.exit()
"""
"""
a = np.array([[[0,0,0],[0,0,1]],[[0,1,0],[0,1,1]]])
b = np.array([[0,0],[0,1],[1,0],[1,1]])

a = np.reshape(a, (-2,3))
"""
[000] -> 00 -> 0
[001] -> 01 -> 1
[010] -> 10 -> 2
[011] -> 11 -> 3
"""
"""
I'm tired
"""
print b
b = b[:, 0]*2 + b[:, 1]

print b
sys.exit()
#b = b.dot(1 << np.arange(b.shape[-1]-1, -1, -1))
#b2 = 2**np.arange(b.shape[1]-1, -1, -1)



print a.shape, b.shape
print a
print b
print ""
#a = np.array([4,5,6,7])
#b = np.array([[0],[1],[2],[3]])
#b = np.array([0,1,2,3])
#print "asdf"
print np.take(a,b, axis=0)

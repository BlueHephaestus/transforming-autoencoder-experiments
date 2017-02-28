import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.int64, shape=[None, 3])
i = tf.placeholder(tf.int64, shape=[None, 2])
y = tf.cast
#y = tf.square(x)
#self.y = tf.placeholder(tf.float32, shape=[None])
#y = tf.gather_nd(x, i)

a = np.array([[[0,0,0],[0,0,1]],[[0,1,0],[0,1,1]]])
b = np.array([[0,0],[0,1],[1,0],[1,1]])

print a.shape, b.shape

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print sess.run(y, feed_dict={x: a, i: b})



import tensorflow as tf

i = tf.constant(0)
#x = tf.placeholder(tf.int64, shape=[1])
x = tf.constant(4)
while_condition = lambda i: tf.less(i, 4)
#b = lambda i: tf.add(x, x)
def b(i):
    tf.add(i, 1)
    return tf.add(x,x)
"""
def body(i):
    #Do something here which you want to do in your loop

    #increment i
    return [tf.add(i, i)]
"""


#do the loop:
r = tf.while_loop(while_condition, b, [i])

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print sess.run(r, feed_dict={})

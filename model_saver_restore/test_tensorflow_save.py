import tensorflow as tf
import numpy as np

# Prepare to feed input, i.e. feed_dict and placeholders
w1 = tf.placeholder("float", name="w1", shape=[2])
w2 = tf.placeholder("float", name="w2", shape=[2])
b1 = tf.Variable(2.0, name="bias")

# Define a test operation that we will restore
y = tf.add(w1, w2, name='add_op')
y_out_prob = tf.nn.softmax(y, name='softmax')
# tf.add_to_collection('pred_network', y)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Create a saver object which will save all the variables
saver = tf.train.Saver()

# Run the operation by feeding input
feed_dict = {w1: [4, 1], w2: [8, 2]}
print(sess.run(y, feed_dict))
print(sess.run(y_out_prob, feed_dict))

# Now, save the graph
saver.save(sess, 'my_test_model', global_step=1)

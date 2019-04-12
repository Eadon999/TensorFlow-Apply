import tensorflow as tf

sess = tf.Session()
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(
    r'E:\virtualboxshare\gitlab\yangxiangdong\bj_user_feature\test\md\my_test_model-1.meta')
saver.restore(sess, tf.train.latest_checkpoint(r'E:\virtualboxshare\gitlab\yangxiangdong\bj_user_feature\test\md'))
# y = tf.get_collection('pred_network')[0]

graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0") # 通过定义网络的placeholder来恢复input
w2 = graph.get_tensor_by_name("w2:0")
y = graph.get_tensor_by_name('add_op:0')
y_prob = graph.get_tensor_by_name('softmax:0')  # 通过定义网络的name来恢复input
y = sess.run(y, feed_dict={w1: [1, 2], w2: [2, 4]})
print(y)
y_prob = sess.run(y_prob, feed_dict={w1: [1, 2], w2: [2, 4]})
print(y_prob)

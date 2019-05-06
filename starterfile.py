import tensorflow as tf

hello = tf.constant('Hello, TensorFlow running in Docker!!')

sess = tf.Session()

print(sess.run(hello))

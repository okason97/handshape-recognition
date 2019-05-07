import tensorflow as tf

hello = tf.constant('Hello, TensorFlow {} running in Docker!!'.format(tf.__version__))

tf.print(hello)

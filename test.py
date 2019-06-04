import tensorflow as tf

def weight_variable_msra(shape, name):
    return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.variance_scaling_initializer(), trainable=True)

def weight_variable_xavier(shape, name):
    return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.xavier_initializer(), trainable=True)

def bias_variable(shape, name = 'bias'):
    initial = tf.constant(0.0, shape = shape)
    return tf.get_variable(name = name, initializer = initial, trainable=True)

current = tf.placeholder("float", shape=[None, 32, 32, 3])
current2 = tf.placeholder("float", shape=[None, 32, 32, 64])
ss = tf.norm(current, ord=1, axis=(1,2))
Ws = weight_variable_xavier([3, 64 ], name = 'W')
bs = bias_variable([ 64 ])

gl = tf.matmul(ss, Ws) + bs
pl_values, pl_indices = tf.nn.top_k(gl, 5)

shape = gl.get_shape()
one_hot = tf.one_hot(pl_indices, shape[1], dtype=tf.float32)
gate = tf.reduce_sum(one_hot, axis=1)

pl = tf.multiply(gl, gate)
''' End '''

current3 = tf.multiply(pl, current2)


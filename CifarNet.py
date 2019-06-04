'''
This is implementation of: Dynamic Channel Pruning Feature Boosting and Suppression
This file implements CifarNet with dynamic layers
'''
# necessary imports
import tensorflow as tf

class CifarNet:
    def __init__(self):
        self.lr = 0.001
        self.label_count = 100

        self.sess = tf.Session()
        
        self.build_model()
        self.sess.run(self.init)

    def train_model(self, batch_X, batch_y):
        feed_dict = {
            self.xs: batch_X,
            self.ys_orig: batch_y,
            self.keep_prob : 1,
            self.is_training : True
        }
        self.sess.run(self.train_step, feed_dict = feed_dict)

        loss = self.sess.run(self.cross_entropy, feed_dict=feed_dict)
        return loss
    
    def test_acc(self, X, y, pruning_rate = 1):
        feed_dict = {
            self.xs: X,
            self.ys_orig: y,
            self.keep_prob : 1,
            self.is_training : True,
            self.pruning_rate : pruning_rate
        }

        acc = self.sess.run(self.accuracy, feed_dict = feed_dict)
        print("Testing accuracy: %f" %acc)


    def build_model(self):
        self.xs = tf.placeholder("float", shape=[None, 32, 32, 3])
        self.ys_orig = tf.placeholder("float", shape=[None, self.label_count])

        self.pruning_rate = tf.placeholder(tf.float32, shape=(1))
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder("bool", shape=[])

        with tf.variable_scope("Conv0", reuse = tf.AUTO_REUSE):
            current = self.batch_activ_conv_gated(self.xs, 3, 64, 3, self.is_training, self.keep_prob)
            print(current)
        with tf.variable_scope("Conv1", reuse = tf.AUTO_REUSE):
            current = self.batch_activ_conv_gated(current, 64, 64, 3, self.is_training, self.keep_prob)
            print(current)
        with tf.variable_scope("Conv2", reuse = tf.AUTO_REUSE):
            current = self.batch_activ_conv_gated(current, 64, 128, 3, self.is_training, self.keep_prob, stride = 2)
            print(current)
        with tf.variable_scope("Conv3", reuse = tf.AUTO_REUSE):
            current = self.batch_activ_conv_gated(current, 128, 128, 3, self.is_training, self.keep_prob)
            print(current)
        with tf.variable_scope("Conv4", reuse = tf.AUTO_REUSE):
            current = self.batch_activ_conv_gated(current, 128, 128, 3, self.is_training, self.keep_prob)
            print(current)
        with tf.variable_scope("Conv5", reuse = tf.AUTO_REUSE):
            current = self.batch_activ_conv_gated(current, 128, 192, 3, self.is_training, self.keep_prob, stride = 2)
            print(current)
        with tf.variable_scope("Conv6", reuse = tf.AUTO_REUSE):
            current = self.batch_activ_conv_gated(current, 192, 192, 3, self.is_training, self.keep_prob)
            print(current)
        with tf.variable_scope("Conv7", reuse = tf.AUTO_REUSE):
            current = self.batch_activ_conv_gated(current, 192, 192, 3, self.is_training, self.keep_prob)
            print(current)
            
            current = self.maxpool2d(current, k=8)
            print(current)
            current = tf.reshape(current, [ -1, 192 ])
            print(current)
        with tf.variable_scope("FC8", reuse = tf.AUTO_REUSE):
            Wfc = self.weight_variable_xavier([ 192, self.label_count ], name = 'W')
            bfc = self.bias_variable([ self.label_count ])
            self.ys_pred = tf.matmul(current, Wfc) + bfc

        self.ys_pred_softmax = tf.nn.softmax(self.ys_pred)
        '''
        Loss Definition
        '''
        # prediction = tf.nn.softmax(ys_)
        # conv_value = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in convValues])
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits = self.ys_pred_softmax, labels = self.ys_orig
        ))
        
        '''
        Optimizer
        '''
        self.train_step = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True).minimize(self.cross_entropy)
        
        '''
        Check whether correct
        '''
        correct_prediction = tf.equal(tf.argmax(self.ys_orig, 1), tf.argmax(self.ys_pred, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        self.init = tf.global_variables_initializer()

    '''
    Helper Builder Functions: to build model more conveniently
    '''
    def weight_variable_msra(self, shape, name):
        return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.variance_scaling_initializer(), trainable=True)

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.xavier_initializer(), trainable=True)

    def bias_variable(self, shape, name = 'bias'):
        initial = tf.constant(0.0, shape = shape)
        return tf.get_variable(name = name, initializer = initial, trainable=True)

    def conv2d(self, input, in_features, out_features, kernel_size, stride=1, with_bias=False):
        W = self.weight_variable_msra([ kernel_size, kernel_size, in_features, out_features ], name = 'kernel')
        conv = tf.nn.conv2d(input, W, [ 1, stride, stride, 1], padding='SAME')
        
        if with_bias:
            return conv + self.bias_variable([ out_features ])
        return conv

    def batch_activ_conv(self, current, in_features, out_features, kernel_size, is_training, keep_prob, stride=1):
        with tf.variable_scope("composite_function", reuse = tf.AUTO_REUSE):
            current = self.conv2d(current, in_features, out_features, kernel_size, stride)
            current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=True)
            # convValues.append(current)
            current = tf.nn.relu(current)
            #current = tf.nn.dropout(current, keep_prob)
        return current

    '''
    This is the implement of dynamic layer
    '''
    def batch_activ_conv_gated(self, current, in_features, out_features, kernel_size, is_training, keep_prob, stride=1):
        with tf.variable_scope("composite_function", reuse = tf.AUTO_REUSE):
            ''' 
            Implement of dynamic layer
            '''
            ss = tf.norm(current, ord=1, axis=(1,2))
            Ws = self.weight_variable_xavier([in_features, out_features ], name = 'W')
            bs = self.bias_variable([ out_features ])

            gl = tf.matmul(ss, Ws) + bs
            length = gl.get_shape().as_list()[-1]
            pl_values, pl_indices = tf.nn.top_k(gl, tf.cast(self.pruning_rate*length, tf.int32))

            shape = gl.get_shape()
            one_hot = tf.one_hot(pl_indices, shape[1], dtype=tf.float32)
            gate = tf.reduce_sum(one_hot, axis=1)
            
            pl = tf.multiply(gl, gate)
            ''' End '''

            current = self.conv2d(current, in_features, out_features, kernel_size, stride)
            current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=True)
            # convValues.append(current)

            current = tf.multiply(pl, current)

            current = tf.nn.relu(current)
            #current = tf.nn.dropout(current, keep_prob)
        return current

    def tf_cov(self, x):
        mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
        mx = tf.matmul(tf.transpose(mean_x), mean_x)
        vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        return cov_xx

    def batch_activ_fc(self, current, in_features, out_features, is_training):
        Wfc = self.weight_variable_xavier([ in_features, out_features ], name = 'W')
        bfc = self.bias_variable([ out_features ])
        current = tf.matmul(current, Wfc) + bfc
        
        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=True)
        current = tf.nn.relu(current)
        return current

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                            padding='VALID')


if __name__ == "__main__":
    model = CifarNet()
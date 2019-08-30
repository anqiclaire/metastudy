import tensorflow as tf


# weight and bais wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initial)

def indexToCoords(w_size, filter_size, index):
    sumOfInc = 0
    realIndex = w_size - (index - 1)
    row = filter_size//2 - 1
    col = filter_size//2 - 1
    for y in range(1, filter_size//2+1):
        sumOfInc += y
        if sumOfInc > realIndex:
            col = y - 1
            row = realIndex - (sumOfInc - y)
            break
    # duplicate coords
    coords = []
    coords.append((row, col))
    coords.append((filter_size-1-row, col))
    coords.append((row, filter_size-1-col))
    coords.append((filter_size-1-row, filter_size-1-col))
    coords.append((col, row))
    coords.append((filter_size-1-col, row))
    coords.append((col, filter_size-1-row))
    coords.append((filter_size-1-col, filter_size-1-row))
    return coords


# # Define network
# # Hidden layer
# num_in_channel = x.get_shape().as_list()[-1]
# num_filters = 8
# filter_size = 10
# w_size = 0
# for i in range(1,filter_size//2+1):
#     w_size += i # w_size: 15
# W = weight_variable(shape=[1, w_size, num_in_channel, num_filters])
# # map weight to bigger weight
# # init fullW
# fullW = tf.Variable(tf.zeros([filter_size, filter_size, num_in_channel, num_filters], tf.float32))
# for nf in range(num_filters):
#     for nc in range(num_in_channel):
#         for index in range(w_size): # 0, 1, 2, ..., 14
#             coords = indexToCoords(w_size, filter_size, index) # a list of coords in fullW that share the same value as W[0, index]
#             for coord in coords:
#                 fullW[coord[0], coord[1], nc, nf].assign(W[0, index, nc, nf])
# TODO
# layer = tf.add(tf.matmul(X, fullW), b)
# conv_layer = tf.nn.relu(z1)

def conv_layer(x, filter_size, num_filters, stride, name):
    """
    Create a 2D convolution layer
    :param x: input from previous layer
    :param filter_size: size of each filter
    :param num_filters: number of filters (or output feature maps)
    :param stride: filter stride
    :param name: layer name
    :return: The output array
    """

    with tf.variable_scope(name):
        num_in_channel = x.get_shape().as_list()[-1]
        num_filters = 8
        filter_size = 10
        w_size = 0
        for i in range(1,filter_size//2+1):
            w_size += i # w_size: 15
        W = weight_variable(shape=[1, w_size, num_in_channel, num_filters])
        # map weight to bigger weight
        # init fullW
        fullW = tf.Variable(tf.zeros([filter_size, filter_size, num_in_channel, num_filters], tf.float32))
        for nf in range(num_filters):
            for nc in range(num_in_channel):
                for index in range(w_size): # 0, 1, 2, ..., 14
                    coords = indexToCoords(w_size, filter_size, index) # a list of coords in fullW that share the same value as W[0, index]
                    for coord in coords:
                        fullW[coord[0], coord[1], nc, nf].assign(W[0, index, nc, nf])
        
        tf.summary.histogram('weight', fullW)
        b = bias_variable(shape=[num_filters])
        tf.summary.histogram('bias', b)
        layer = tf.nn.conv2d(x, fullW,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        layer += b
        tf.add_to_collection('conv_w', W)
        conv_layer = tf.nn.relu(layer)
        tf.add_to_collection('layer', layer)
        tf.add_to_collection('conv_layer', conv_layer)
        return conv_layer

def fc_layer(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    with tf.variable_scope(name):
        in_dim = x.get_shape()[1]
        W = weight_variable(shape=[in_dim, num_units])
        tf.summary.histogram('weight', W)
        b = bias_variable(shape=[num_units])
        tf.summary.histogram('bias', b)
        layer = tf.matmul(x, W)
        layer += b
        if use_relu:
            layer = tf.nn.relu(layer)
        # if use_tanh:
        #     layer = tf.math.tanh(layer)
        print(layer.shape)
        return layer


def flatten_layer(layer):
    """
    Flattens the output of the convolutional layer to be fed into fully-connected layer
    :param layer: input array
    :return: flattened array
    """
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


def max_pool(x, ksize, stride, name):
    """
    Create a max pooling layer
    :param x: input to max-pooling layer
    :param ksize: size of the max-pooling filter
    :param stride: stride of the max-pooling filter
    :param name: layer name
    :return: The output array
    """
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding="SAME",
                          name=name)
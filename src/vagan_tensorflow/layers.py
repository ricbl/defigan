# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)
# Modified by Ricardo Bigolin Lanfredi 
# Last modified: 2020-07-01

import tensorflow as tf
import numpy as np
import logging

from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
STANDARD_NONLINEARITY = tf.nn.relu

def maxpool3D_layer(x, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="SAME"):
    '''
    nets3D max pooling layer with 2x2x2 pooling as default
    '''

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], kernel_size[2], 1]
    strides_aug = [1, strides[0], strides[1], strides[2], 1]

    op = tf.nn.max_pool3d(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op

def averagepool3D_layer(x, name=None):
    '''
    nets3D max pooling layer with 2x2x2 pooling as default
    '''

    op = tf.reduce_mean(x, axis=(1,2,3), keep_dims=False, name=name)
    tf.summary.histogram(op.op.name + '/activations', op)

    return op

def crop_and_concat_layer(inputs, axis=-1):

    '''
    Layer for cropping and stacking feature maps of different size along a different axis. 
    Currently, the first feature map in the inputs list defines the output size. 
    The feature maps can have different numbers of channels. 
    :param inputs: A list of input tensors of the same dimensionality but can have different sizes
    :param axis: Axis along which to concatentate the inputs
    :return: The concatentated feature map tensor
    '''

    output_size = inputs[0].get_shape().as_list()[1:]
    # output_size = tf.shape(inputs[0])[1:]
    concat_inputs = [inputs[0]]

    for ii in range(1,len(inputs)):

        larger_size = inputs[ii].get_shape().as_list()[1:]
        # larger_size = tf.shape(inputs[ii])

        # Don't subtract over batch_size because it may be None
        start_crop = np.subtract(larger_size, output_size) // 2

        if len(output_size) == 4:  # nets3D images
        # if output_size.shape[0] == 5:  # nets3D images
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[0], start_crop[1], start_crop[2], 0),
                                     (-1, output_size[0], output_size[1], output_size[2], -1))
        elif len(output_size) == 3:  # nets2D images
        # elif output_size.shape[0] == 4:
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[0], start_crop[1], 0),
                                     (-1, output_size[0], output_size[1], -1))
        else:
            raise ValueError('Unexpected number of dimensions on tensor: %d' % len(output_size))

        concat_inputs.append(cropped_tensor)

    return tf.concat(concat_inputs, axis=axis)
#
def batch_normalisation_layer(bottom, name, training, moving_average_decay=0.99):
    '''
    Wrapper for tensorflows own batch normalisation function. 
    :param bottom: Input layer (should be before activation)
    :param name: A name for the computational graph
    :param training: A tf.bool specifying if the layer is executed at training or testing time
    :return: Batch normalised activation
    '''

    h_bn = tf.contrib.layers.batch_norm(inputs=bottom,
                                        decay=moving_average_decay,
                                        epsilon=1e-3,
                                        is_training=training,
                                        scope=name,
                                        center=True,
                                        scale=True)

    return h_bn


### FEED_FORWARD LAYERS ##############################################################################33

def conv3D_layer(bottom,
                 name,
                 kernel_size=(3,3,3),
                 num_filters=32,
                 strides=(1,1,1),
                 activation=STANDARD_NONLINEARITY,
                 padding="SAME",
                 weight_init='he_normal',
                 add_bias=True):

    '''
    Standard nets3D convolutional layer
    '''

    bottom_num_filters = bottom.get_shape().as_list()[-1]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)
        op = tf.nn.conv3d(bottom, filter=weights, strides=strides_augm, padding=padding)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op

def deconv3D_layer(bottom,
                   name,
                   kernel_size=(4,4,4),
                   num_filters=32,
                   strides=(2,2,2),
                   output_shape=None,
                   activation=STANDARD_NONLINEARITY,
                   padding="SAME",
                   weight_init='he_normal',
                   add_bias=True):

    '''
    Standard nets2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a
    factor of 2. 
    '''

    bottom_shape = bottom.get_shape().as_list()

    if output_shape is None:
        batch_size = tf.shape(bottom)[0]
        output_shape = tf.stack([batch_size, bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], bottom_shape[3]*strides[2], num_filters])

    bottom_num_filters = bottom_shape[4]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], num_filters, bottom_num_filters]

    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)

        op = tf.nn.conv3d_transpose(bottom,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides_augm,
                                    padding=padding)

        # op = tf.reshape(op, output_shape)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op

### BATCH_NORM SHORTCUTS #####################################################################################

def conv3D_layer_bn(bottom,
                    name,
                    training,
                    kernel_size=(3,3,3),
                    num_filters=32,
                    strides=(1,1,1),
                    activation=STANDARD_NONLINEARITY,
                    padding="SAME",
                    weight_init='he_normal',
                    bn_momentum=0.99,
                    init_weights=None,
                    init_biases=None):

    '''
    Shortcut for batch normalised nets3D convolutional layer
    '''

    with tf.variable_scope(name):

        conv = conv3D_layer(bottom=bottom,
                            name=name + '_conv',
                            kernel_size=kernel_size,
                            num_filters=num_filters,
                            strides=strides,
                            activation=tf.identity,
                            padding=padding,
                            weight_init=weight_init,
                            add_bias=False)

        conv_bn = batch_normalisation_layer(conv, name + '_bn', training, moving_average_decay=bn_momentum)

        act = activation(conv_bn)

    return act

def deconv3D_layer_bn(bottom,
                      name,
                      training,
                      kernel_size=(4,4,4),
                      num_filters=32,
                      strides=(2,2,2),
                      output_shape=None,
                      activation=STANDARD_NONLINEARITY,
                      padding="SAME",
                      weight_init='he_normal',
                      bn_momentum=0.99):

    '''
    Shortcut for batch normalised nets3D transposed convolutional layer
    '''

    deco = deconv3D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=tf.identity,
                          padding=padding,
                          weight_init=weight_init,
                          add_bias=False)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training, moving_average_decay=bn_momentum)

    act = activation(deco_bn)

    return act

### VARIABLE INITIALISERS ####################################################################################

def get_weight_variable(shape, name=None, type='xavier_uniform', regularize=True, **kwargs):

    if 'init_weights' in kwargs and kwargs['init_weights'] is not None:
        type = 'pretrained'
        logging.info('Using pretrained weights for layer: %s' % name)

    initialise_from_constant = False
    if type == 'xavier_uniform':
        initial = xavier_initializer(uniform=True, dtype=tf.float32)
    elif type == 'xavier_normal':
        initial = xavier_initializer(uniform=False, dtype=tf.float32)
    elif type == 'he_normal':
        initial = variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'he_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'caffe_uniform':
        initial = variance_scaling_initializer(uniform=True, factor=1.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'simple':
        stddev = kwargs.get('stddev', 0.02)
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        initialise_from_constant = True
    elif type == 'bilinear':
        weights = _bilinear_upsample_weights(shape)
        initial = tf.constant(weights, shape=shape, dtype=tf.float32)
        initialise_from_constant = True
    elif type == 'pretrained':
        initial = kwargs.get('init_weights')
        initialise_from_constant = True
    else:
        raise ValueError('Unknown initialisation requested: %s' % type)

    if name is None:  # This keeps to option open to use unnamed Variables
        weight = tf.Variable(initial)
    else:
        if initialise_from_constant:
            weight = tf.get_variable(name, initializer=initial)
        else:
            weight = tf.get_variable(name, shape=shape, initializer=initial)

    if regularize:
        tf.add_to_collection('weight_variables', weight)

    return weight



def get_bias_variable(shape, name=None, init_value=0.0, **kwargs):

    if 'init_biases' in kwargs and kwargs['init_biases'] is not None:
        initial = kwargs['init_biases']
        logging.info('Using pretrained weights for layer: %s' % name)
    else:
        initial = tf.constant(init_value, shape=shape, dtype=tf.float32)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)



def _upsample_filt(size):
    '''
    Make a nets2D bilinear kernel suitable for upsampling of the given (h, w) size.
    '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def _bilinear_upsample_weights(shape):
    '''
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    '''

    if not shape[0] == shape[1]: raise ValueError('kernel is not square')
    if not shape[2] == shape[3]: raise ValueError('input and output featuremaps must have the same size')

    kernel_size = shape[0]
    num_feature_maps = shape[2]

    weights = np.zeros(shape, dtype=np.float32)
    upsample_kernel = _upsample_filt(kernel_size)

    for i in range(num_feature_maps):
        weights[:, :, i, i] = upsample_kernel

    return weights

def _add_summaries(op, weights, biases):

    # Tensorboard variables
    tf.summary.histogram(weights.name, weights)
    if biases:
        tf.summary.histogram(biases.name, biases)
    tf.summary.histogram(op.op.name + '/activations', op)
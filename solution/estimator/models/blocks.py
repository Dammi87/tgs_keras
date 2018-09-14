"""Block methods.

CREDIT: https://github.com/qubvel/segmentation_models
"""
import tensorflow as tf
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Lambda
import keras.backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec


def conform_a2b(a, b, up_rate=(1, 1)):
    """Conform size of to B."""
    a_shape = K.int_shape(a)
    b_shape = K.int_shape(b)
    h_diff = b_shape[1] - a_shape[1] * up_rate[0]
    w_diff = b_shape[2] - a_shape[2] * up_rate[1]
    print(a_shape, b_shape)
    print(h_diff, w_diff)
    # Pad before upsample
    if h_diff > 0 or w_diff > 0:
        up = int(h_diff / 2)
        down = h_diff - up
        left = int(w_diff / 2)
        right = w_diff - left

        def pad_layer(x):
            """Wrap pad fcn."""
            return K.spatial_2d_padding(x, padding=((up, down), (left, right)), data_format='channels_last')
        a = Lambda(pad_layer)(a)

    return a


def upsample_concatenate(x, skip, up_layer, up_rate=(2, 2)):
    """Concatenate, but pad if needed."""
    # Conform if needed
    x = conform_a2b(x, skip, up_rate)

    # Now upsample
    x = up_layer(x)

    # Conform if needed
    skip = conform_a2b(skip, x)
    # And concatenate
    x = Concatenate(axis=-1)([x, skip])  # Channels first

    return x


def handle_block_names(stage):
    """Create name for the blocks given the certain stage."""
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def conv_relu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    """Return a layer that does conv+bn+acticvation."""
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x
    return layer


def upsample2d_block(filters, stage, kernel_size=(3, 3), upsample_rate=(2, 2), use_batchnorm=False, skip=None):
    """Return a layer that does simple upsampling."""
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)
        up_layer = UpSampling2D(size=upsample_rate, name=up_name)

        if skip is not None:
            x = upsample_concatenate(input_tensor, skip, up_layer, up_rate=upsample_rate)
        else:
            x = up_layer(input_tensor)

        x = conv_relu(filters,
                      kernel_size,
                      use_batchnorm=use_batchnorm,
                      conv_name=conv_name + '1',
                      bn_name=bn_name + '1',
                      relu_name=relu_name + '1')(x)
        x = conv_relu(filters,
                      kernel_size,
                      use_batchnorm=use_batchnorm,
                      conv_name=conv_name + '2',
                      bn_name=bn_name + '2',
                      relu_name=relu_name + '2')(x)
        return x
    return layer


def transpose2d_block(filters, stage, kernel_size=(3, 3), upsample_rate=(2, 2), transpose_kernel_size=(4, 4), use_batchnorm=False, skip=None):
    """Return a layer that does transposed convolution as upsampling method."""
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        def up_layer(x):
            """Wrap upsampling layer."""
            x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate, padding='same',
                                name=up_name, use_bias=not(use_batchnorm))(x)
            if use_batchnorm:
                x = BatchNormalization(name=bn_name + '1')(x)
            x = Activation('relu', name=relu_name + '1')(x)

            return x

        if skip is not None:
            x = upsample_concatenate(input_tensor, skip, up_layer, up_rate=upsample_rate)
        else:
            x = up_layer(input_tensor)

        x = conv_relu(filters, kernel_size, use_batchnorm=use_batchnorm, conv_name=conv_name +
                      '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x
    return layer


def upsample_billinear(x, scale):
    """Scale x with scale."""
    if scale == 1:
        return x
    x_shape = K.int_shape(x)
    size = (x_shape[1] * scale, x_shape[2] * scale)
    y_shape = (x_shape[0], x_shape[1] * scale, x_shape[2] * scale, x_shape[-1])

    def resize(x_unscaled):
        return tf.image.resize_bilinear(x_unscaled, size)

    return Lambda(resize, name='billinear_upsample_%d' % scale)(x)


def downsample_billinear(x, scale):
    """Scale x with scale."""
    if scale == 1:
        return x
    x_shape = K.int_shape(x)
    size = (int(x_shape[1] / scale), int(x_shape[2] / scale))

    def resize(x_unscaled):
        return tf.image.resize_bilinear(x_unscaled, size)

    return Lambda(resize, name='billinear_downsample_%d' % scale)(x)


def ReflectionPadding2D(x, padding):
    """Pad using reflection."""
    # Get shape
    x_shape = K.int_shape(x)
    # Calculate outut shape
    comb = [sum(pad) for pad in padding[1:]]

    out_shape = [a + b for a, b in zip(comb, x_shape[1:])]

    def call(x):
        """Apply."""
        return tf.pad(x, tf.constant(padding), "REFLECT")

    return Lambda(call, output_shape=out_shape)(x)

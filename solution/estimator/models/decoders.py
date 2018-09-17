"""Initialize models."""
from keras.layers import Conv2D, Cropping2D, Conv2DTranspose, SpatialDropout2D
from keras.layers import Activation, Concatenate, Dropout, BatchNormalization
from keras.models import Model
import keras.backend as K
from .blocks import transpose2d_block, upsample2d_block, upsample_billinear, downsample_billinear, residual_block
from .blocks import upsample_concatenate_like


def print_info(x, skip_connection):
    """Print info about connections."""
    print("Building a model, skip connections are")
    for skip in skip_connection:
        print("\t{}: {}".format(skip.name, skip.shape))
    print("Final output from encoder")
    print("\t{}: {}".format(x.name, x.shape))


def regular(encoder, skip_connection, _input,
            block_type='upsampling',
            activation='elu',
            use_dropout=False,
            use_batchnorm=False,
            hyper_concat=False):
    """Create a decoder given encoder, decoder ensures 2x always.

    Parameters
    ----------
    encoder : TYPE
        Description
    skip_connection : TYPE
        Description
    _input : TYPE
        Description
    upsample_rates : tuple, optional
        How much to upsample by
    decoder_filters : tuple, optional
        Depth of the decoder
    block_type : str, optional
        Upsampling strategy
    activation : str, optional
        Output activation
    use_batchnorm : bool, optional
        Use batchnorm or not in decoder
    hyper_concat : bool, optional
        Description

    Returns
    -------
    Model
        The Model

    Raises
    ------
    NotImplementedError
        Description
    """
    x = encoder.output

    if block_type == 'transpose':
        up_type = transpose2d_block
    else:
        up_type = upsample2d_block

    # Middle layer
    out_channels = K.int_shape(x)[-1]
    x = Conv2D(int(out_channels), (3, 3), padding='same', name='middle_layer')(x)
    x = Activation(activation, name='mid_activation')(x)

    # Upsample
    resized = []
    for i in range(len(skip_connection)):
        upsample_rate = (2, 2)
        x = up_type(out_channels, i, upsample_rate=upsample_rate,
                    skip=skip_connection[i], use_batchnorm=use_batchnorm, activation=activation)(x)
        x = SpatialDropout2D(0.2)(x) if use_dropout else x
        resized.append(x)
        out_channels = int(out_channels / 2)

    if hyper_concat:
        hyper_conv = []
        n_up = len(resized)
        for i, layer in enumerate(resized):
            scale = 2 ** (n_up - i - 1)
            hyper_conv.append(upsample_billinear(layer, scale))
        x = Concatenate(axis=-1)(hyper_conv)
        logits = Conv2D(1, (3, 3), padding='same', name='hyper_logits')(x)
    else:
        # I split the last convlayer to be different when hyper-concat is active, allowing to transfer learn
        logits = Conv2D(1, (3, 3), padding='same', name='logits')(x)

    # Now, resize and de-pad the output
    this_shape = K.int_shape(logits)

    # Check if de-pad then resize, or simply depad
    if this_shape[1] == 128:
        # De-pad only
        logits = Cropping2D(cropping=((14, 13), (14, 13)), input_shape=this_shape)(logits)
    elif this_shape[1] == 101:
        # Pass
        pass
    else:
        raise NotImplementedError

    logit_sigmoid = Activation('sigmoid', name='logit_sigmoid')(logits)

    out = Concatenate(axis=-1, name='logit_and_sigmoid')([logits, logit_sigmoid])
    model = Model(_input, out)

    return model


def simple101(encoder, skip_connection, _input,
              block_type='upsampling',
              activation='elu',
              use_dropout=False,
              use_batchnorm=False,
              hyper_concat=False):
    """Implement Hordur's decoder, capable of outputing 101x101.

    Parameters
    ----------
    encoder : TYPE
        Description
    skip_connection : TYPE
        Description
    _input : TYPE
        Description
    upsample_rates : tuple, optional
        Description
    decoder_filters : tuple, optional
        Description
    block_type : str, optional
        Description
    activation : str, optional
        Description
    use_batchnorm : bool, optional
        Description
    hyper_concat : bool, optional
        Description
    """

    # Get input
    x = encoder.output
    filters = int(K.int_shape(x)[-1] / 2)

    # Check if 101 should be forced (Then valid needs to be used sometimes in transpose..)
    input_shape = K.int_shape(encoder.input)

    def up_block(n_filters, x, skip, pad):
        x = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding=pad, kernel_initializer="he_normal")(x)
        x = Concatenate()([x] + skip)
        x = Dropout(0.5)(x)

        x = Conv2D(n_filters, (3, 3), activation=None, padding="same", kernel_initializer="he_normal")(x)
        x = residual_block(x, n_filters)
        return residual_block(x, n_filters, True)

    # Start upsampling
    for i, skip in enumerate(skip_connection):
        pad = 'same'
        if input_shape[1] == 101 and (i == 1 or i == 3):
            pad = 'valid'
        x = up_block(filters, x, skip, pad)
        filters = int(filters / 2)

    logits = Conv2D(1, (1, 1), padding="same", activation=None)(x)
    logit_sigmoid = Activation('sigmoid', name='logit_sigmoid')(logits)

    out = Concatenate(axis=-1, name='logit_and_sigmoid')([logits, logit_sigmoid])
    model = Model(_input, out)
    return model


def resnet(encoder, skip_connection, _input,
           block_type='upsampling',
           activation='elu',
           use_dropout=False,
           use_batchnorm=False,
           hyper_concat=False):
    """Decider resnet encoder, will end up with 101x101 given a 202x202 input.

    Parameters
    ----------
    encoder : TYPE
        Description
    skip_connection : TYPE
        Description
    _input : TYPE
        Description
    block_type : str, optional
        Description
    activation : str, optional
        Description
    use_batchnorm : bool, optional
        Description
    hyper_concat : bool, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    x = encoder.output
    filters = [512, 256, 128, 64]

    def up_block(n_filters, x, skip):
        x = upsample_concatenate_like(x, skip)
        for _ in range(2):
            x = Conv2D(n_filters, (3, 3), activation=None, padding="same", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x) if use_batchnorm else x
            x = Activation(activation)(x)
            x = SpatialDropout2D(0.25)(x) if use_dropout else x
        return x

    # Start upsampling
    for i, skip in enumerate(skip_connection):
        x = up_block(filters[i], x, skip)

    # Output
    logits = Conv2D(1, (3, 3), padding="same", activation=None)(x)
    logit_sigmoid = Activation('sigmoid', name='logit_sigmoid')(logits)

    out = Concatenate(axis=-1, name='logit_and_sigmoid')([logits, logit_sigmoid])
    model = Model(_input, out)
    return model

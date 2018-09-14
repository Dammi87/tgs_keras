"""Initialize models."""
from keras.layers import Conv2D, Cropping2D
from keras.layers import Activation, Concatenate
from keras.models import Model
from keras.utils import plot_model
import keras.backend as K

from .encoders import get_model
from .blocks import transpose2d_block, upsample2d_block, upsample_billinear, downsample_billinear


def print_info(x, skip_connection):
    """Print info about connections."""
    print("Building a model, skip connections are")
    for skip in skip_connection:
        print("\t{}: {}".format(skip.name, skip.shape))
    print("Final output from encoder")
    print("\t{}: {}".format(x.name, x.shape))


def build(encoder_type, classes=1,
          upsample_rates=(2, 2, 2, 2, 2),
          decoder_filters=(512, 256, 128, 64, 32),
          block_type='upsampling',
          freeze_encoder=False,
          activation='sigmoid',
          use_batchnorm=False,
          hyper_concat=False):
    """Build a model using the desired encoder.

    Parameters
    ----------
    encoder_type : str
        One of ['vgg16' 'vgg19' 'res50' 'inceptionv3' 'InceptionResNetV2' 'densenet_121' 'densenet_169' 'densenet_201']
    classes : int, optional
        Number of output classes
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

    Returns
    -------
    Model
        The Model
    """
    encoder, skip_connection, _input = get_model(encoder_type)

    if freeze_encoder:
        for layer in encoder.layers:
            layer.trainable = False

    x = encoder.output

    if block_type == 'transpose':
        up_block = transpose2d_block
    else:
        up_block = upsample2d_block

    # Middle layer
    out_channels = x.shape[-1]
    x = Conv2D(int(out_channels * 2), (3, 3), padding='same', name='middle_layer')(x)
    x = Activation('relu', name='mid_activation')(x)
    # Upsample
    resized = []
    for i in range(len(skip_connection)):
        upsample_rate = (upsample_rates[i], upsample_rates[i])
        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection[i], use_batchnorm=use_batchnorm)(x)
        resized.append(x)

    if hyper_concat:
        hyper_conv = []
        n_up = len(resized)
        for i, layer in enumerate(resized):
            scale = 2 ** (n_up - i - 1)
            hyper_conv.append(upsample_billinear(layer, scale))
        x = Concatenate(axis=-1)(hyper_conv)
        logits = Conv2D(classes, (3, 3), padding='same', name='hyper_logits')(x)
    else:
        # I split the last convlayer to be different when hyper-concat is active, allowing to transfer learn
        logits = Conv2D(classes, (3, 3), padding='same', name='logits')(x)

    # Now, resize and de-pad the output
    this_shape = K.int_shape(logits)

    # Check if de-pad then resize, or simply depad
    if this_shape[1] == 128:
        # De-pad only
        logits = Cropping2D(cropping=((14, 13), (14, 13)), input_shape=this_shape)(logits)
    elif this_shape[1] == 228:
        # Depad then resize
        logits = Cropping2D(cropping=((13, 13), (13, 13)), input_shape=this_shape)(logits)
        logits = downsample_billinear(logits, 2)
    else:
        raise NotImplementedError

    logit_sigmoid = Activation(activation, name='logit_sigmoid')(logits)

    out = Concatenate(axis=-1, name='logit_and_sigmoid')([logits, logit_sigmoid])
    model = Model(_input, out)

    return model


if __name__ == "__main__":
    mdl = build('vgg16', hyper_concat=False)
    mdl.summary()
    plot_model(mdl, to_file='model.png')

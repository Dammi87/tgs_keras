"""Initialize models."""
from keras.utils import plot_model
from .encoders import get_model
from .decoders import regular, simple101, resnet


def build(encoder_type, decoder_type,
          freeze_encoder=False,
          use_dropout=False,
          n_filters=512,
          activation='elu',
          use_batchnorm=False):
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
    print()
    encoder, skip_connection, _input = get_model(encoder_type)

    if freeze_encoder:
        for layer in encoder.layers:
            layer.trainable = False

    if decoder_type == 'regular':
        return regular(encoder, skip_connection, _input,
                       activation=activation,
                       use_batchnorm=use_batchnorm,
                       use_dropout=use_dropout,
                       hyper_concat=False)

    elif decoder_type == 'simple101':
        return simple101(encoder, skip_connection, _input,
                         activation=activation,
                         use_batchnorm=use_batchnorm,
                         use_dropout=use_dropout,
                         hyper_concat=False)

    elif decoder_type == 'resnet':
        return resnet(encoder, skip_connection, _input,
                      activation=activation,
                      use_batchnorm=use_batchnorm,
                      use_dropout=use_dropout,
                      hyper_concat=False)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    mdl = build(encoder_type='simple101', decoder_type='simple101')
    mdl.summary()
    plot_model(mdl, to_file='model.png')

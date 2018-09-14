"""Methods to create the custom models."""
from .resnet_32 import create_model as resnet_32_create_model


def get_custom_model(mdl_name):
    """Get custom encoder."""
    if mdl_name == 'resnet_32_relu':
        def encoder(include_top, weights, input_tensor, input_shape, classes):
            """Create resnet 32 with Relu and BN."""
            return resnet_32_create_model(input_shape, name='resnet_32_relu', activation='relu')

    elif mdl_name == 'resnet_32_elu':
        def encoder(include_top, weights, input_tensor, input_shape, classes):
            """Create resnet 32 with Relu and BN."""
            return resnet_32_create_model(input_shape, name='resnet_32_elu', activation='elu')

    else:
        raise NotImplementedError

    return encoder

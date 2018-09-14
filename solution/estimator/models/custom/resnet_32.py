"""ResNet-32 model for CIFAR-10 image recognition.

Inspired by:
    Deep Learning Specialization at Coursera
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
Model architecture for CIFAR-10 dataset classification reproduced from original paper:
    https://arxiv.org/pdf/1512.03385.pdf
Karolis M. December 2017.
"""
from keras.layers import Conv2D, BatchNormalization, Activation, add, Input, GlobalAveragePooling2D
from keras.models import Model


def building_block(x, activation, filter_size, filters, stride=1):
    """Resnet building block."""
    # Save the input value for shortcut
    x_shortcut = x

    # Reshape shortcut for later adding if dimensions change
    if stride > 1:

        x_shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(x_shortcut)
        x_shortcut = BatchNormalization(axis=3)(x_shortcut) if activation == 'relu' else x_shortcut

    # First layer of the block
    x = Conv2D(filters, kernel_size=filter_size, strides=stride, padding='same')(x)
    x = BatchNormalization(axis=3)(x) if activation == 'relu' else x
    x = Activation(activation)(x)

    # Second layer of the block
    x = Conv2D(filters, kernel_size=filter_size, strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x) if activation == 'relu' else x
    x = add([x, x_shortcut])  # Add shortcut value to main path
    x = Activation(activation)(x)

    return x


def create_model(input_shape, name, activation='relu'):
    """Build the model."""
    # Define the input
    x_input = Input(input_shape)

    # Stage 1
    x = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same')(x_input)
    x = BatchNormalization(axis=3)(x) if activation == 'relu' else x
    x = Activation(activation)(x)

    # Stage 2
    x = building_block(x, activation, filter_size=3, filters=16, stride=1)
    x = building_block(x, activation, filter_size=3, filters=16, stride=1)
    x = building_block(x, activation, filter_size=3, filters=16, stride=1)
    x = building_block(x, activation, filter_size=3, filters=16, stride=1)
    x = building_block(x, activation, filter_size=3, filters=16, stride=1)

    # Stage 3
    x = building_block(x, activation, filter_size=3, filters=32, stride=2)  # dimensions change (stride=2)
    x = building_block(x, activation, filter_size=3, filters=32, stride=1)
    x = building_block(x, activation, filter_size=3, filters=32, stride=1)
    x = building_block(x, activation, filter_size=3, filters=32, stride=1)
    x = building_block(x, activation, filter_size=3, filters=32, stride=1)

    # Stage 4
    x = building_block(x, activation, filter_size=3, filters=64, stride=2)  # dimensions change (stride=2)
    x = building_block(x, activation, filter_size=3, filters=64, stride=1)
    x = building_block(x, activation, filter_size=3, filters=64, stride=1)
    x = building_block(x, activation, filter_size=3, filters=64, stride=1)
    x = building_block(x, activation, filter_size=3, filters=64, stride=1)

    # Average pooling and output layer
    x = GlobalAveragePooling2D()(x)

    return Model(inputs=x_input, outputs=x, name=name)


if __name__ == "__main__":
    mdl = create_model((3, 256, 256), 'relu')
    mdl.summary()

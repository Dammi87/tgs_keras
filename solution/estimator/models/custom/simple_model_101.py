"""Create the simple model."""
from keras.layers import Input, Lambda, Add, Conv2D
from keras import Model
import keras.backend as K
from solution.estimator.models.blocks import max_pool, simple_101_down_block, residual_block


def mdl(input_layer, filters):
    """Hordur baby."""
    # 101 -> 50
    conv1 = simple_101_down_block(input_layer, filters * 1)
    pool1 = max_pool(conv1, .25)

    # 50 -> 25
    conv2 = simple_101_down_block(pool1, filters * 2)
    pool2 = max_pool(conv2, .25)

    # 25 -> 12
    conv3 = simple_101_down_block(pool2, filters * 4)
    pool3 = max_pool(conv3, .25)

    # 12 -> 6
    conv4 = simple_101_down_block(pool3, filters * 8)
    pool4 = max_pool(conv4, .25)

    # encoder 2
    # 101 -> 50
    conv1_1 = simple_101_down_block(input_layer, filters * 1)
    pool1_1 = max_pool(conv1_1, .1)

    # 50 -> 25
    conv2_1 = simple_101_down_block(pool1_1, filters * 2)
    pool2_1 = max_pool(conv2_1, .15)

    # 25 -> 12
    conv3_1 = simple_101_down_block(pool2_1, filters * 4)
    pool3_1 = max_pool(conv3_1, 0.2)

    # 12 -> 6
    conv4_1 = simple_101_down_block(pool3_1, filters * 8)
    pool4_1 = max_pool(conv4_1, 0.2)

    # encoder 3
    # 101 -> 50
    conv1_2 = simple_101_down_block(input_layer, filters * 1)
    pool1_2 = max_pool(conv1_2, 0.15)

    # 50 -> 25
    conv2_2 = simple_101_down_block(pool1_2, filters * 2)
    pool2_2 = max_pool(conv2_2, 0.2)

    # 25 -> 12
    conv3_2 = simple_101_down_block(pool2_2, filters * 4)
    pool3_2 = max_pool(conv3_2, 0.15)

    # 12 -> 6
    conv4_2 = simple_101_down_block(pool3_2, filters * 8)
    pool4_2 = max_pool(conv4_2, 0.25)

    concat = Add()([pool4_2, pool4_1, pool4])

    convm = Conv2D(filters * 16, (3, 3), activation=None, padding="same")(concat)
    convm = residual_block(convm, filters * 16)
    convm = residual_block(convm, filters * 16, True)

    skips = [[conv4, conv4_1, conv4_2], [conv3, conv3_1, conv3_2], [conv2, conv2_1, conv2_2], [conv1, conv1_1, conv1_2]]
    return convm, skips


def simple_model_101():
    """Return decoder 101."""
    # The input chain is rather hard-coded for 3 channel, simply remove the rest
    _input = Input(shape=(101, 101, 3), name='original_image')

    def get_layer(ch):
        def lambda_fcn(y_pred):
            return K.reshape(y_pred[:, :, :, ch], (-1, 101, 101, 1))
        return lambda_fcn

    new_input = Lambda(get_layer(0))(_input)

    outputs, skip_layers = mdl(new_input, 16)
    model = Model(_input, outputs)
    return model, skip_layers, _input

from solution.estimator.models import build

mdl = build('simple101', 'simple101',
            freeze_encoder=False,
            use_dropout=True,
            activation='relu',
            use_batchnorm=True)

mdl.summary()

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Concatenate
from keras.layers import UpSampling2D, Input, Dropout, Activation, Add, Conv2DTranspose
from keras import Model

def mdl(input_layer, filters):

    # encoder 1
    # 101 -> 50
    conv1 = down_block(input_layer, filters * 1)
    pool1 = max_pool(conv1, .25)

    # 50 -> 25
    conv2 = down_block(pool1, filters * 2)
    pool2 = max_pool(conv2, .25)

    # 25 -> 12
    conv3 = down_block(pool2, filters * 4)
    pool3 = max_pool(conv3, .25)

    # 12 -> 6
    conv4 = down_block(pool3, filters * 8)
    pool4 = max_pool(conv4, .25)

    #encoder 2
    # 101 -> 50
    conv1_1 = down_block(input_layer, filters * 1)
    pool1_1 = max_pool(conv1_1, .1)

    # 50 -> 25
    conv2_1 = down_block(pool1_1, filters * 2)
    pool2_1 = max_pool(conv2_1, .15)

    # 25 -> 12
    conv3_1 = down_block(pool2_1, filters * 4)
    pool3_1 = max_pool(conv3_1, 0.2)

    # 12 -> 6
    conv4_1 = down_block(pool3_1, filters * 8)
    pool4_1 = max_pool(conv4_1, 0.2)

    #encoder 3
    # 101 -> 50
    conv1_2 = down_block(input_layer, filters * 1)
    pool1_2 = max_pool(conv1_2, 0.15)

    # 50 -> 25
    conv2_2 = down_block(pool1_2, filters * 2)
    pool2_2 = max_pool(conv2_2, 0.2)

    # 25 -> 12
    conv3_2 = down_block(pool2_2, filters * 4)
    pool3_2 = max_pool(conv3_2, 0.15)

    # 12 -> 6
    conv4_2 = down_block(pool3_2, filters * 8)
    pool4_2 = max_pool(conv4_2, 0.25)

    concat = Add()([pool4_2, pool4_1, pool4])


    # Middle
    convm = Conv2D(filters * 16, (3, 3), activation=None, padding="same")(concat)
    convm = residual_block(convm, filters * 16)
    convm = residual_block(convm, filters * 16, True)

    # decoder
    # 6 -> 12
    deconv4 = Conv2DTranspose(filters * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = Concatenate()([deconv4, conv4, conv4_1, conv4_2])
    uconv4 = Dropout(0.6)(uconv4)

    uconv4 = Conv2D(filters * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, filters * 8)
    uconv4 = residual_block(uconv4, filters * 8, True)

    # 12 -> 25 (padding=valid -> +1)
    deconv3 = Conv2DTranspose(filters * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = Concatenate()([deconv3, conv3, conv3_1, conv3_2])
    uconv3 = Dropout(0.55)(uconv3)

    uconv3 = Conv2D(filters * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, filters * 4)
    uconv3 = residual_block(uconv3, filters * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = Concatenate()([deconv2, conv2, conv2_1, conv2_2])
    uconv2 = Dropout(0.5)(uconv2)

    uconv2 = Conv2D(filters * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, filters * 2)
    uconv2 = residual_block(uconv2, filters * 2, True)

    # 50 -> 101 (padding=valid -> +1)
    deconv1 = Conv2DTranspose(filters * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = Concatenate()([deconv1, conv1, conv1_1, conv1_2])

    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(filters * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, filters * 1)
    uconv1 = residual_block(uconv1, filters * 1, True)

    output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  Activation('sigmoid')(output_layer_noActi)

    return output_layer

def simple_model_101():
    inputs = Input((101, 101, 1))
    outputs = mdl(inputs, 16)
    model = Model(inputs, outputs)
    return model

def down_block(inputs, filters):
    x = Conv2D(filters, (3, 3), activation=None, padding="same", kernel_initializer="he_normal")(inputs)
    x = residual_block(x, filters)
    x = residual_block(x, filters, True)
    return x

def max_pool(inputs, dropout=.5):
    x = MaxPooling2D((2, 2))(inputs)
    x = Dropout(dropout)(x)
    return x

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, size, strides=strides, padding=padding, kernel_initializer="he_normal")(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, filters, (3,3) )
    x = convolution_block(x, filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

mdl = simple_model_101()
mdl.summary()
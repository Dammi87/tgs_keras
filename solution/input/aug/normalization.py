"""Contains normalization settings to use."""
import numpy as np
from keras.applications.vgg16 import preprocess_input as vgg16_pre
from keras.applications.vgg19 import preprocess_input as vgg19_pre
from keras.applications.resnet50 import preprocess_input as resnet50_pre

VGG_MEAN = np.array([123.68, 116.779, 103.939]).reshape(1, 1, 1, 3).astype(np.float32)


def normalize_01(img):
    """Normalize the input image to be between 0-1."""
    return img.astype(np.float32) / 255


def normalize_vgg16(img):
    """Normalize the input image using the vgg16 mean."""
    return vgg16_pre(img)


def normalize_vgg19(img):
    """Normalize the input image using the vgg19 mean."""
    return vgg19_pre(img)


def normalize_resnet50(img):
    """Normalize the input image using the resnet50 mean."""
    return resnet50_pre(img)


# ['vgg16' 'vgg19' 'res50' 'inceptionv3' 'InceptionResNetV2' 'densenet_121' 'densenet_169' 'densenet_201']
defined = {
    'basic': {
        'img': normalize_01,
        'mask': normalize_01,
        'img_channels': 1
    },
    'vgg16': {
        'img': normalize_01,
        'mask': normalize_01,
        'img_channels': 3
    },
    'vgg19': {
        'img': normalize_01,
        'mask': normalize_01,
        'img_channels': 3
    }
}


def get_norm_method(network_type):
    """Fetch normalization methods."""
    if network_type not in defined:
        network_type = 'basic'

    return defined[network_type]

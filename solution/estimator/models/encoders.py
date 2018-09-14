"""VGG16 Header pre-trained."""
import keras.applications as applications
from keras.layers import Input, Lambda
# from solution.estimator.models.custom import get_custom_model
from .blocks import ReflectionPadding2D, upsample_billinear


def resnet_up_pad(img_shape):
    """Upsample, then pad."""
    def up_pad(x):
        print("Upsampling the: ", x)
        x = upsample_billinear(x, scale=2)
        print("\tResult: ", x)
        x = ReflectionPadding2D(x, padding=[[0, 0], [11, 11], [11, 11], [0, 0]])
        print("\tResult: ", x)
        return x
    return up_pad


def get_header(header):
    """Return the header."""
    skip_layer_names = []
    skip_layer_idx = []
    pad_layer = None
    img_shape = None
    if header == 'vgg16':
        fcn = applications.vgg16.VGG16
        skip_layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
        img_shape = (128, 128, 3)
        pad_layer = lambda x : ReflectionPadding2D(x, padding=[[0, 0], [14, 13], [14, 13], [0, 0]])
    elif header == 'vgg19':
        fcn = applications.vgg19.VGG19
        skip_layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']
        img_shape = (128, 128, 3)
        pad_layer = lambda x: ReflectionPadding2D(x, padding=[[0, 0], [14, 13], [14, 13], [0, 0]])
    elif header == 'res32_relu':
        # fcn = get_custom_model(header)
        skip_layer_names = ['activation_11', 'activation_21']
    elif header == 'res50':
        fcn = applications.resnet50.ResNet50
        skip_layer_names = ['activation_1', 'activation_10', 'activation_22', 'activation_40']
        img_shape = (228, 228, 3)
        pad_layer = resnet_up_pad(img_shape)
    elif header == 'inceptionv3':
        # 299x299 for input!
        fcn = applications.inception_v3.InceptionV3
        skip_layer_idx = [228, 86, 16, 9]
    elif header == 'InceptionResNetV2':
        fcn = applications.inception_resnet_v2.InceptionResNetV2
        skip_layer_idx = [594, 260, 16, 9]
    elif header == 'densenet_121':
        fcn = applications.densenet.DenseNet121
        skip_layer_idx = [311, 139, 51, 4]
    elif header == 'densenet_169':
        fcn = applications.densenet.DenseNet169
        skip_layer_idx = [367, 139, 51, 4]
    elif header == 'densenet_201':
        fcn = applications.densenet.DenseNet201
        skip_layer_idx = [479, 139, 51, 4]
    else:
        raise NotImplementedError

    def build_model():
        img_input = Input(shape=(101, 101, 3), name='original_image')  # Always input original size

        if pad_layer is not None:
            enc = pad_layer(img_input)
        else:
            enc = Lambda(lambda x: x, output_shape=(128, 128, 3))(img_input)

        # pre_process_model = Model(img_input, enc)

        # First build encoder
        encoder = fcn(include_top=False,
                      weights='imagenet',
                      input_tensor=enc,
                      input_shape=img_shape,
                      classes=None)

        # Remove last
        skip_layers = []
        if len(skip_layer_idx) == 0:
            skip_layers = [encoder.get_layer(layer).get_output_at(0) for layer in skip_layer_names]
        else:
            skip_layers = [encoder.layers[layer].get_output_at(0) for layer in skip_layer_idx]

        return encoder, list(reversed(skip_layers)), img_input

    return build_model


def get_model(model_type):
    """Get the correct model."""
    mdl, skip_layers, img_input = get_header(model_type)()

    return mdl, skip_layers, img_input


if __name__ == "__main__":
    encoder, skip_layers, _input = get_model('res50')
    encoder.summary()
    print([(l.name, l.shape) for l in skip_layers])

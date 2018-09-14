"""Contains available augmentations, which are only applied to images."""
from imgaug import augmenters as iaa
import cv2
import numpy as np


intensity_seq = iaa.Sequential([
    # iaa.Invert(0.3),
    iaa.OneOf([
        iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5)))
    ]),
    iaa.OneOf([
        iaa.Noop(),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.AverageBlur(k=(2, 2)),
            iaa.MedianBlur(k=(3, 3))
        ])
    ])
], random_order=False)


def _combination(images, random_state, parents, hooks):
    new_img = []
    for img in images:
        smoothness = np.absolute(cv2.Laplacian(img.copy(), cv2.CV_64F)).astype(np.uint8)
        edges = cv2.Canny(img.copy(), 150, 150).astype(np.uint8)
        new_img.append(np.stack([smoothness, img.squeeze(), edges], axis=-1))

    return np.stack(new_img)


def _reflective_padding_128(image, random_state, parents, hooks):
    new_img = []
    for img in image:
        new_img.append(cv2.copyMakeBorder(img, 14, 13, 14, 13, cv2.BORDER_REFLECT))

    return np.stack(new_img)


def _reflective_padding_228(image, random_state, parents, hooks):
    new_img = []
    for img in image:
        double = cv2.resize(image, (202, 202), interpolation=cv2.INTER_CUBIC)
        new_img.append(cv2.copyMakeBorder(double, 13, 13, 13, 13, cv2.BORDER_REFLECT))

    return np.stack(new_img)


channel_augmenter = iaa.Lambda(func_images=_combination, func_keypoints=None)
reflect_resize_128 = iaa.Lambda(func_images=_reflective_padding_128, func_keypoints=None)
reflect_resize_228 = iaa.Lambda(func_images=_reflective_padding_228, func_keypoints=None)


# Collect all into a dictionary
defined = {
    'intensity_seq': intensity_seq,
    'channel_augmenter': channel_augmenter,
    'resize_128': reflect_resize_128,
    'resize_228': reflect_resize_228}

"""Contains available augmentations, which are applied to both masks and images deterministicially."""
import cv2
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia

crop_flip = iaa.Sequential([
    # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Crop(px=(0, 8)),
    # horizontally flip 50% of the images
    iaa.Fliplr(0.5),
])


def _perspective_transform_augment_images(self, images, random_state, parents, hooks):
    result = images
    if not self.keep_size:
        result = list(result)

    matrices, max_heights, max_widths = self._create_matrices(
        [image.shape for image in images],
        random_state
    )

    for i, (M, max_height, max_width) in enumerate(zip(matrices, max_heights, max_widths)):
        warped = cv2.warpPerspective(images[i], M, (max_width, max_height))
        if warped.ndim == 2 and images[i].ndim == 3:
            warped = np.expand_dims(warped, 2)
        if self.keep_size:
            h, w = images[i].shape[0:2]
            warped = ia.imresize_single_image(warped, (h, w))

        result[i] = warped

    return result


iaa.PerspectiveTransform._augment_images = _perspective_transform_augment_images

affine = iaa.Sequential([
    # Deformations
    iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.029, 0.03))),
    iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.059, 0.069))),
], random_order=True)


# Collect all into a dictionary
defined = {
    'crop_flip': crop_flip,
    'affine': affine
}

"""Classes and methods for dataset boostrapping and augmentations."""
from imgaug import augmenters as iaa
import numpy as np
from . import aug
from solution.lib.neptune import Params


class ImageProcessor:
    """A class that holds all augmentations required for each images."""

    def __init__(self):
        """Initialize class."""
        self._pre_process = []
        self._images_only = []
        self._masks_only = []
        self._mask_and_images = []
        self._is_setup = False

    def add_pre_process(self, iaa_method):
        """Add a iaa method to the preprocessing of the images."""
        self._pre_process.append(iaa_method)

    def add_images_only(self, iaa_method):
        """Add a iaa method that should be applied for image_converterping."""
        self._images_only.append(iaa_method)

    def add_masks_only(self, iaa_method):
        """Add a iaa method that should be applied for masks_converterping."""
        self._masks_only.append(iaa_method)

    def add_both(self, iaa_method):
        """Add a iaa method that should be applied on both masks and images."""
        self._mask_and_images.append(iaa_method)

    def set_normalization_method(self, method_name):
        """Set the normalization method."""
        self._normalize_method = method_name

    def _setup(self):
        """Setup the augmentation chain."""
        if self._is_setup:
            return

        # Only setup once
        self._is_setup = True

        methods = [
            '_pre_process',
            '_images_only',
            '_masks_only',
            '_mask_and_images'
        ]

        for method in methods:
            attr = getattr(self, method)
            if len(attr) == 0:
                new = iaa.Noop()
            elif len(attr) == 1:
                new = attr[0]
            else:
                new = iaa.Sequential(attr)

            if method == '_mask_and_images':
                new = new.to_deterministic()

            setattr(self, method, new)

        # Change out normalization
        self.apply_image_normalization = self._normalize_method['img']
        self.apply_mask_normalization = self._normalize_method['mask']
        self.output_image_channels = self._normalize_method['img_channels']

    def apply_preprocess(self, img, batch_size=10):
        """Apply image preprocessing steps."""
        self._setup()

        # Loop to save RAM
        index = np.arange(img.shape[0])
        new_img = []
        for index in range(0, img.shape[0], batch_size):
            batch = img[index:min(index + batch_size, img.shape[0]), ::]
            processed = self._pre_process.augment_images(batch)
            new_img.append(processed)

        return np.concatenate(new_img, axis=0)

    def apply(self, img, mask):
        """Apply the augmentations to the image and mask.

        NOTE: Images and masks should NOT be normalized!!

        Parameters
        ----------
        img : TYPE
            Description
        mask : None, optional
            Description

        Returns
        -------
        img : np.ndarray
            The images with augmentations, normalized between 0 and 1
        mask : np.ndarray
            The mask with augmentations, binary 0 and 1
        """
        self._setup()

        # Apply both
        img = self._mask_and_images.augment_images(img)
        mask = self._mask_and_images.augment_images(mask)

        # Apply image only step
        img = self._images_only.augment_images(img)

        # Apply mask only step
        mask = self._masks_only.augment_images(mask)

        return self.apply_image_normalization(img), self.apply_mask_normalization(mask)

    def apply_normalization(self, img, mask):
        """Normalize images."""
        return self.apply_image_normalization(img), self.apply_mask_normalization(mask)


def get_processor():
    """Load an augmenter class for training / inference.

    Returns
    -------
    ImageProcessor
        A class that will apply the desired pre-processing  / augmentations of images
        Main methods are
            .apply(img, mask) # Mask is optional
            .apply_preprocess(img) # If preprocessing is needed, it's done here. This should
                                     also be done for the test set.

    """
    # Instantiate
    augmenter = ImageProcessor()

    # Fetch what augmentations are requested. Augmentations start with aug_SOME_NAME
    params = Params().get_dict()

    for param in params:
        if 'aug_' in param:

            # If false, dont go further
            if not params[param]:
                continue

            # Check if this augmentation exists and add it
            aug_name = param.replace('aug_', '')
            if aug_name in aug.pre_process and params[param]:
                augmenter.add_pre_process(aug.pre_process[aug_name])
            elif aug_name in aug.image:
                augmenter.add_images_only(aug.image[aug_name])
            elif aug_name in aug.mask:
                augmenter.add_masks_only(aug.mask[aug_name])
            elif aug_name in aug.image_and_masks:
                augmenter.add_both(aug.image_and_masks[aug_name])

            print("Applied %s" % param)

    # Apply normalization method
    augmenter.set_normalization_method(aug.get_norm_method(params['model_type']))

    return augmenter

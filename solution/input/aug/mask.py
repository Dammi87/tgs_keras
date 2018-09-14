"""Mask conversions."""
from imgaug import augmenters as iaa
import cv2
import numpy as np
import scipy.ndimage as ndimage

def get_smoothed_mask(img, w0=255, sigma=8, dilate=10):
    # Calculate the contours
    img_empty = 255 * np.ones(img.shape, dtype=np.uint8)
    # Dilate the mask
    mask_dil = cv2.erode(img.copy(), kernel=np.ones((dilate, dilate), np.uint8), iterations = 1)
    _, contours, _ = cv2.findContours(mask_dil.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Return if no contours:
    if len(contours) == 0:
        return img

    all_dil_contours_together = cv2.drawContours(img_empty.copy(), contours, -1, (0, 0), -1)

    # Create an empty container to fill in
    distance_map = np.zeros(list(img.shape) + [len(contours)])

    # Loop through all contours
    a = []
    for i, i_contour in enumerate(contours):
        this_contour = cv2.drawContours(img_empty.copy(), [i_contour], -1, (0, 0), 0)
        edt = ndimage.distance_transform_edt(this_contour == 255)
        distance_map[:, :, i] = edt
        a.append(this_contour)

    weight = w0 * np.exp(-np.divide(np.power(np.min(distance_map, axis=-1), 2), 2 * sigma**2))

    # Load into the weighted mask, the original mask at the points of dilution
    loc = all_dil_contours_together == 0
    weight[loc] = img[loc]
    # This is for debugging only, will show original contour
    # _, original_contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # all_contours_together = cv2.drawContours(img_empty.copy(), original_contours, -1, (0, 0), 0)
    # weight[all_contours_together == 0] = 0

    return weight


def _smooth_wrapper(images, random_state, parents, hooks):
    new_img = []
    for img in images:
        if img.sum() == 0:
            new_img.append(img)
            continue
        img = np.expand_dims(get_smoothed_mask(img.squeeze()), 2)
        new_img.append(img)

    return np.array(new_img)

mask_smoother_augmenter = iaa.Lambda(func_images=_smooth_wrapper, func_keypoints=None)


# Collect all into a dictionary
defined = {'mask_smooth': mask_smoother_augmenter}
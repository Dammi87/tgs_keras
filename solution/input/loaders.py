"""Methods to load the dataset."""
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import os
import numpy as np


class DatasetLoader():
    """Dataset loader."""

    def __init__(self,
                 path,
                 img_ch=1,
                 remove_blank_images=False,
                 remove_bad_masks=False,
                 resize=128,
                 split=0.1,
                 kfold=None,
                 kfold_nbr=0,
                 post_df_methods=[],
                 load_test_set=False):
        """Initialize the loader class.

        Parameters
        ----------
        path : str
            Path to the downloaded kaggle dataset
        resize : int, optional
            Specify the size to which the images should be resized to, None keeps
            the same size. Default is 128
        split : float, optional
            Validation ratio, default is 0.1
        post_df_methods : None, optional
            A list of methods to apply after dataframes have been loaded.
            The dataframes have the following columns
                images: Numpy arrays with the loaded images in the specified size, normalized
                masks: Numpy arrays with the loaded masks in the specified size, normalized
                coverage: Indicator of how much salts there is in each iamge
                z: The depthf of each image
        load_test_set : bool, optional
            Loads the test set instead of the training set
        """
        self._path = path
        self._img_ch = img_ch
        self._resize = None
        self._post_df_methods = post_df_methods
        self._load_test_set = load_test_set
        self._split = split
        self._kfold = kfold
        self._kfold_nbr = kfold_nbr
        self._remove_blank_images = remove_blank_images and not load_test_set
        self._remove_bad_masks = remove_bad_masks and not load_test_set

        # Load the dataset with the properties specified
        self._load()

    def _load(self):
        train_df = pd.read_csv(os.path.join(self._path, "train.csv"), index_col="id", usecols=[0])
        depths_df = pd.read_csv(os.path.join(self._path, "depths.csv"), index_col="id")
        train_df = train_df.join(depths_df)

        bw = self._img_ch != 3
        # Load test set
        if self._load_test_set:
            pd.options.mode.chained_assignment = None
            df = depths_df[~depths_df.index.isin(train_df.index)]

            img_paths = os.path.join(self._path, 'images')
            df["images"] = [load_img(os.path.join(img_paths, "{}.png".format(idx)), self._resize, bw=bw)
                            for idx in df.index]
            df["images_org"] = [load_img(os.path.join(img_paths, "{}.png".format(idx)), None, bw=bw)
                                for idx in df.index]
        else:
            df = train_df
            img_paths = os.path.join(self._path, "train", 'images')
            mask_paths = os.path.join(self._path, "train", "masks")
            df["images"] = [load_img(os.path.join(img_paths, "{}.png".format(idx)), self._resize, bw=bw)
                            for idx in df.index]
            df["masks"] = [load_mask(os.path.join(mask_paths, "{}.png".format(idx)), self._resize) for idx in df.index]
            df["masks_org"] = [load_mask(os.path.join(mask_paths, "{}.png".format(idx)), None) for idx in df.index]
            df["images_org"] = [load_img(os.path.join(img_paths, "{}.png".format(idx)), None, bw=bw)
                                for idx in df.index]

            # Salt coverage
            img_size = df["images"][0].shape[1]
            df["coverage"] = df.masks.map(np.sum) / pow(img_size, 2)

        # Apply the post methods
        for method in self._post_df_methods:
            df = method(df)

        # Save
        self._df = df

    def _get_train_and_validation(self):
        """Split the dataframe into training and validation.

        Parameters
        ----------
        split : float, optional
            Ratio of the validation set

        Returns
        -------
        train: tuple
            A tuple containing two lists, where the first one is numpy arrays for images, and the
            second one is numpy array for masks
        validation: tuple
            A tuple containing two lists, where the first one is numpy arrays for images, and the
            second one is numpy array for masks
        """
        def cov_to_class(val):
            for i in range(0, 11):
                if (val / 255) / 10 <= i:
                    return i

        # Apply coverage binning for stratification
        self._df["coverage_class"] = self._df.coverage.map(cov_to_class)

        if self._kfold is None:
            ids_train, ids_valid = get_stratified(self._df, self._split)
        else:
            ids_train, ids_valid = get_kfold(self._df, self._kfold, self._kfold_nbr)

        # Remove bad id's
        if self._remove_blank_images:
            ids_train = [_id for _id in ids_train if _id not in BLANK_IMAGES]
        if self._remove_bad_masks:
            ids_train = [_id for _id in ids_train if _id not in BAD_MASKS]

        train = {
            'x': np.array([self._df.images[img] for img in ids_train]),
            'y': np.array([self._df.masks[img] for img in ids_train]),
            'id': ids_train
        }
        valid = {
            'x': np.array([self._df.images[img] for img in ids_valid]),
            'y': np.array([self._df.masks[img] for img in ids_valid]),
            'id': ids_valid
        }

        return [train, valid]

    def _get_test(self):
        """Return the images from the test set."""
        return [{'x': np.array(self._df.images.tolist()),
                 'y': None,
                 'x_org': np.array(self._df.images_org.tolist()),
                 'id': self._df.index}]

    def get_dataset(self):
        """Get the dataset."""
        if self._load_test_set:
            return self._get_test()
        else:
            return self._get_train_and_validation()


def load_img(img_path, resize=None, bw=True):
    """Load an image as a numpy array normalized, resize if specified.

    Parameters
    ----------
    img_path : TYPE
        Path to the image to be loaded
    resize : None, int
        Resize images to this width and height

    Returns
    -------
    np.ndarray, shape (width, height, 1) or (resize, resize, 1)
        A numpy array containing the pixel values for the image, normalized by 255
    """
    if bw:
        img = np.array(resize_img(Image.open(img_path).convert('L'), resize))
        return np.expand_dims(img, 2)
    else:
        return np.array(resize_img(Image.open(img_path), resize))


def load_mask(img_path, resize=None):
    """Load a mask as a numpy array, resize if specified.

    Parameters
    ----------
    img_path : TYPE
        Path to the image to be loaded
    resize : None, int
        Resize images to this width and height

    Returns
    -------
    np.ndarray, shape (width, height, 1) or (resize, resize, 1)
        A numpy array containing the pixel values for the mask, True value is salt
    """
    mask = np.array(resize_img(Image.open(img_path).convert('L'), resize, method=Image.NEAREST))
    return np.expand_dims(mask, 2)


def resize_img(img, resize, method=Image.BILINEAR):
    """Resize images using the specified method."""
    if resize:
        img = img.resize((resize, resize), method)
    return img


def get_stratified(df, test_size):
    """Return a stratified split of the dataset."""
    ids_train, ids_valid = train_test_split(df.index.values,
                                            test_size=test_size,
                                            stratify=df.coverage_class,
                                            random_state=1337)
    return ids_train, ids_valid


def get_kfold(df, folds, get_nr):
    """Return a kfold, return a specific fold."""
    kf = KFold(n_splits=folds, random_state=1337, shuffle=True)
    ids_train, ids_valid = list(kf.split(df.index.values))[get_nr]

    return ids_train, ids_valid


"""Misc data."""
BLANK_IMAGES = ['1f0b16aa13',
                '5edb37f5a8',
                'a536f382ec',
                '97515a958d',
                '99909324ed',
                'a9e940dccd',
                '8c1d0929a2',
                '96049af037',
                '1efe1909ed',
                '5aa0015d15',
                'a3e0a0c779',
                'c20069b110',
                'd6437d0c25',
                'e7da2d7800',
                '2fb6791298',
                'b552fb0d9d',
                '5ff89814f5',
                'd1665744c3',
                'a31e485287',
                'e0da89ce88',
                'dcca025cc6',
                '37df75f3a2',
                '6b95bc6c5f',
                'a8be31a3c1',
                'f26e6cffd6',
                '1f73caa937',
                'e82421363e',
                '1c0b2ceb2f',
                'bedb558d15',
                'd0e720b57b',
                'd93d713c55',
                'b9bf0422a6',
                'c3589905df',
                'e51599adb5',
                '0d8ed16206',
                '40ccdfe09d',
                '20ed65cbf8',
                'a56e87840f',
                '1b0d74b359',
                'b637a7621a',
                '96d1d6138a',
                '58789490d6',
                '7769e240f0',
                'd8bed49320',
                '9260b4f758',
                'd2e14828d5',
                '808cbefd71',
                'ff9d2e9ba7',
                '6f79e6d54b',
                'a2b7af2907',
                '287b0f197f',
                '51870e8500',
                '8ee20f502e',
                'a48b9989ac',
                'b11110b854',
                'cc15d94784',
                'd0244d6c38',
                '590f7ae6e7',
                '573f9f58c4',
                'aa97ecda8e',
                'b8c3ca0fab',
                'ec542d0719',
                '423ae1a09c',
                '10833853b3',
                '1a7f8bd454',
                'c1c6a1ebad',
                'acb95dd7c9',
                'f9fc7746fb',
                '4f30a97219',
                'a9fd8e2a06',
                'f0190fc4b4',
                '9aa65d393a',
                '135ae076e9',
                'f2c869e655',
                '3ee4de57f8',
                '755c1e849f',
                '762f01c185',
                'c8404c2d4f',
                '05b69f83bf',
                '3ff3881428',
                ]

BAD_MASKS = ['a266a2a9df',
             '50d3073821',
             'd4d34af4f7',
             '7845115d01',
             '7deaf30c4a',
             'b525824dfc',
             '483b35d589',
             'f0c401b64b',
             '849881c690',
             'a9ee40cf0d',
             'c83d9529bd',
             '52667992f8',
             'ba1287cb48',
             'aeba5383e4',
             'caccd6708f',
             'cb4f7abe67',
             'd0bbe4fd97',
             '9b29ca561d',
             '3cb59a4fdc',
             'b24d3673e1',
             'f6e87c1458',
             '80a458a2b6',
             '6bc4c91c27',
             'c387a012fc',
             '09b9330300',
             '8367b54eac',
             '95f6e2b2d1',
             '09152018c4',
             'be18a24c49',
             '458a05615e',
             '23afbccfb5',
             '58de316918',
             '62aad7556c',
             '24522ec665',
             'c2973c16f1',
             '96523f824a',
             'f641699848',
             '0280deb8ae',
             'd9a52dc263',
             '50b3aef4c4',
             '39cd06da7d',
             '52ac7bb4c1',
             '7c0b76979f',
             'd4d2ed6bd2',
             'be7014887d',
             'f75842e215',
             'dd6a04d456',
             '5bffe3b52f',
             '62d30854d7',
             '33887a0ae7',
             '6fe4017680',
             'c27409a765',
             '130229ec15',
             '7f0825a2f0',
             '1eaf42beee',
             'e73ed6e7f2',
             '876e6423e6',
             'fb3392fee0',
             'b35b1b412b',
             '0b45bde756',
             '4f5df40ab2',
             'f7380099f6',
             '9a4b15919d',
             'ddcb457a07',
             'b8a9602e21',
             '87afd4b1ca',
             'f19b7d20bb',
             'e6e3e58c43',
             '5b217529e7',
             '182bfc6862',
             'baac3469ae',
             '4fdc882e4b',
             '33dfce3a76',
             'a6625b8937',
             '834861f1b6',
             '2f746f8726',
             '5f98029612',
             '7b5d5d40fe',
             '3975043a11',
             '06d21d76c4',
             'b63b23fdc9',
             'bfbb9b9149',
             '6460ce2df7',
             '4fbda008c7',
             '81fa3d59b8',
             '285f4b2e82',
             '9067effd34',
             '96216dae3b',
             'b1be1fa682',
             '88a5c49514',
             '608567ed23',
             '99ee31b5bc',
             '71f7425387',
             '90720e8172',
             '2bc179b78c',
             '937ea43a65',
             '916aff36ae',
             '56f4bcc716',
             '00950d1627',
             'febd1d2a67',
             '403cb8f4b3',
             '4b72e35b8b',
             '4ef0559016',
             '9eb4a10b98',
             'bfa7ee102e',
             '93a1541218',
             '640ceb328a',
             'b7b83447c4',
             'c98dfd50ba',
             'de7202d286',
             'be90ab3e56',
             '15d76f1672',
             'cef03959d8',
             '53e17edd83',
             'e12cd094a6',
             '49336bb17b',
             'ad2fa649f7',
             'fb47e8e74e',
             '919bc0e2ba'
             ]

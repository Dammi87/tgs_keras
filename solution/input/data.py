"""Data methods."""
import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self,
                 is_training,
                 is_test,
                 dataset,
                 batch_size=32,
                 shuffle=True,
                 processor=None):
        """Init class."""
        self.is_training = is_training
        self.is_test = is_test
        self.batch_size = batch_size
        self.dataset = dataset
        self.id = dataset['id']
        self.masks = dataset['y']
        self.images = dataset['x']
        self.list_ids = list(range(len(self.id)))
        self.indexes = np.arange(len(self.list_ids))
        self.shuffle = shuffle
        self.processor = processor
        self.on_epoch_end()
        self.pre()

    def pre(self):
        """Apply potential pre-processing step."""
        if self.processor:
            self.images = self.processor.apply_preprocess(self.images)

    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate a batch."""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        """Update index after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_tmp):
        """Generate data containing batch_size number of examples.

        Parameters
        ----------
        list_ids_tmp : TYPE
            Description

        Returns
        -------
        X : (n_samples, h, w, n_channels)
        y : (n_samples, h, w)
        """
        # If not test set, get both img and masks
        if not self.is_test:
            x = np.take(self.images, list_ids_tmp, axis=0)
            y = np.take(self.masks, list_ids_tmp, axis=0)
        else:
            x = np.take(self.images, list_ids_tmp, axis=0)

        # Apply processor if available.
        if self.processor:
            # Now, if we are testing only, apply normalization and return
            if self.is_test:
                return self.processor.apply_image_normalization(x)

            # If we are training, apply augmentations and return
            if self.is_training:
                x, y = self.processor.apply(x, y)
                return x, y

            # If we make it here, we are validating, only normalize.
            x, y = self.processor.apply_normalization(x, y)
            return x, y

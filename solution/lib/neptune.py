"""Contains methods that help with neptune related stuff."""
from keras.callbacks import Callback
from deepsense import neptune
from argparse import Namespace
import numpy as np
import warnings


class Params():
    """Only query context once."""

    the_one = None

    def __new__(cls):
        """Initialize as a singleton."""
        if cls.the_one is None:
            cls.the_one = super(Params, cls).__new__(cls)
            params = get_params(neptune.Context(), as_dict=True)
            for key in params:
                setattr(cls.the_one, key, params[key])
            cls.the_one._dict = params

        return cls.the_one

    def get_dict(self):
        """Return parameters as dictionary."""
        return self._dict

    def set_param(self, param, value):
        """Overwrite a parameter with the value."""
        setattr(self, param, value)


def get_params(ctx, as_dict=False):
    """Fetch the neptune parameters, "None" are set to None."""
    new_args = {}
    for key in ctx.params:
        if ctx.params[key] == "None":
            new_args[key] = None
        else:
            new_args[key] = ctx.params[key]

    # Add job_id
    new_args['job_id'] = ctx.experiment_id

    if as_dict:
        return new_args

    return Namespace(**new_args)


class ModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self._ctx = neptune.Context()

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        # Sending the monitor metric
                        print('Sending x={}, y={} on {}'.format(epoch, self.best, 'val_iou_kaggle_metric'))
                        self._ctx.channel_send('val_iou_kaggle_metric', x=epoch, y=self.best)

                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))

                    # Get iou and send
                    current = logs.get('val_loss')
                    print('Sending x={}, y={} on {}'.format(epoch, current, 'val_loss'))
                    self._ctx.channel_send('val_loss', x=epoch, y=current)
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

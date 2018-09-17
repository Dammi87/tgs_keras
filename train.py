"""Methods to build the training from the Neptune params."""
import os
import numpy as np
from keras.models import Model, Input, model_from_json
from keras.layers import Average, Activation, Lambda
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, TerminateOnNaN
from keras import optimizers
import keras.backend as K
from solution.lib.neptune import Params, ModelCheckpoint
from solution.estimator.models import build
from solution.estimator import loss
from solution.estimator import metric
from solution.input import data, bootstrap, loaders
import solution.lib.kaggle as kaggle

# TODO
# [X] Make sure the predictions from logits -> predictions are correct when using cross-entropy and not
#     - Fixed, simply output raw logits, loss functions will take care of converting to sigmoid
# [X] Check and see what binary_crossentropy in Keras uses (logits or sigmoid?)
#     - They use sigmoid!
# [ ] Create SWA + Cyclic Learning rate as fine-tune
# [ ] Split up helper functions
# [ ] Add CRF
# [ ] Create method to add depth information to images


params = Params()


def get_json_file(job_id):
    """Path to json file."""
    model_dir = get_model_dir(job_id)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return os.path.join(model_dir, '%s.json' % job_id)


def get_encoder_decoder():
    """Destruct string into encoder-decoder parts."""
    encoder_decoder = params.encoder_decoder
    encoder, decoder = encoder_decoder.split('_')
    return encoder, decoder


def get_submission_file(job_id):
    """Given experiment ID, return the model_dir."""
    model_dir = os.path.join(params.model_dir, params.encoder_decoder, job_id, 'submission.csv')
    return model_dir


def get_model_dir(job_id):
    """Given experiment ID, return the model_dir."""
    model_dir = os.path.join(params.model_dir, params.encoder_decoder, job_id, 'chkp')
    return model_dir


def get_log_dir(job_id):
    """Given experiment ID, return the log_dir."""
    log_dir = os.path.join(params.model_dir, params.encoder_decoder, job_id, 'logs')
    return log_dir


def get_model_chkp(job_id):
    """Given experiment ID, return the weight file."""
    file_path = os.path.join(get_model_dir(job_id), "weights.best.hdf5")
    return file_path


def maybe_resume(mdl, experiment_id=None):
    """Maybe resume from a chkp."""
    resume_from = params.resume_from if experiment_id is None else experiment_id
    file_path = None
    # First resume from takes president
    if resume_from:
        file_path = get_model_chkp(resume_from)

    # But if there is already a checkpoint in job_id dir, then use that
    if os.path.isfile(get_model_chkp(params.job_id)):
        file_path = get_model_chkp(params.job_id)

    if file_path is not None:
        print("Loading weights from %s" % file_path)
        mdl.load_weights(file_path, by_name=True)

    return mdl


def get_metric(scan=False):
    """Return the metric used for training."""
    if scan:
        return metric.iou_kaggle_metric_scan()

    if params.monitor_metric == 'iou_kaggle_metric':
        # Need to wrap to split the prediction into two
        def iou_kaggle_metric(mask, y_true):
            prediction = Lambda(lambda x: x[:, :, :, 1])(y_true)
            return metric.iou_kaggle_metric(mask, prediction)

        return [iou_kaggle_metric]

    return 'accuracy'


def get_callbacks():
    """Get all callbacks for training."""
    model_dir = get_model_dir(params.job_id)
    log_dir = get_log_dir(params.job_id)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save checkpoints at
    file_path = get_model_chkp(params.job_id)
    checkpoint = ModelCheckpoint(file_path, monitor='val_iou_kaggle_metric', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    callbacks_list = [checkpoint, TerminateOnNaN()]

    # Check for rest
    if params.early_stopping:
        callbacks_list.append(EarlyStopping(patience=params.early_stopping_patience,
                                            verbose=1, monitor='val_iou_kaggle_metric', mode='max'))

    if params.reduce_lr_on_plateau:
        callbacks_list.append(ReduceLROnPlateau(factor=params.rlop_factor, monitor='iou_kaggle_metric', mode='max',
                                                patience=params.rlop_patience, min_lr=params.rlop_min_lr, verbose=1))

    if params.tensorboard:
        callbacks_list.append(TensorBoard(log_dir=log_dir,
                                          histogram_freq=0,
                                          batch_size=32,
                                          write_graph=True,
                                          write_grads=False,
                                          write_images=False,
                                          embeddings_freq=0,
                                          embeddings_layer_names=None,
                                          embeddings_metadata=None,
                                          embeddings_data=None))

    return callbacks_list


def get_optimizer():
    """Get desired optimizer."""
    return getattr(optimizers, params.optimizer)(lr=params.learning_rate)


def get_loss():
    """Build the loss method."""
    print("Generating loss function")
    loss_list = []

    if params.loss_dice_weight > 0.001:
        def wrapped_dice_weighted(masks, predictions, logits):
            return params.loss_dice_weight * loss.dice_loss(masks, predictions)

        loss_list.append(wrapped_dice_weighted)
        print("\tDice loss * {}".format(params.loss_dice_weight))

    if params.loss_bce_weight > 0.001:
        def wrapped_bce_weighted(masks, predictions, logits):
            return params.loss_bce_weight * loss.binary_xentropy(masks, predictions)

        loss_list.append(wrapped_bce_weighted)
        print("\tBCE loss * {}".format(params.loss_bce_weight))
    if params.focal_loss_weight > 0.001:
        def wrapped_focal_weighted(masks, predictions, logits):
            return params.focal_loss_weight * loss.focal_loss(params.focal_loss_gamma, params.focal_loss_alpha)(masks, predictions)

        loss_list.append(wrapped_focal_weighted)
        print("\tFocal loss * {}".format(params.focal_loss_weight))

    if params.loss_lovaz_weight > 0.001:
        def wrapped_lovazs_weighted(masks, predictions, logits):
            return params.loss_lovaz_weight * loss.keras_lovasz_hinge(masks, logits)

        loss_list.append(wrapped_lovazs_weighted)
        print("\tLovaz loss * {}".format(params.loss_lovaz_weight))

    def final_loss(masks, y_pred):
        _loss = 0

        def get_layer(ch):
            def lambda_fcn(y_pred):
                return K.reshape(y_pred[:, :, :, ch], (-1, 101, 101, 1))
            return lambda_fcn

        logits = Lambda(get_layer(0))(y_pred)
        predictions = Lambda(get_layer(1))(y_pred)
        for fcn in loss_list:
            _loss += fcn(masks, predictions, logits)
        return _loss

    return final_loss


def save_model_json(mdl, experiment_id):
    """Save model to json file."""
    json_path = get_json_file(experiment_id)
    json_str = mdl.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(json_str)

    print("Saved json file to %s" % json_path)


def get_model(experiment_id=None, freeze_encoder=True, compile_mdl=True):
    """Get the model."""
    enc, dec = get_encoder_decoder()
    mdl = build(enc, dec,
                freeze_encoder=freeze_encoder,
                use_dropout=params.use_dropout,
                activation=params.decoder_activation,
                use_batchnorm=True if params.use_batchnorm and params.decoder_activation != 'elu' else False)
    # Save json
    mdl.to_json
    mdl = maybe_resume(mdl, experiment_id)
    mdl.compile(loss=get_loss(), optimizer=get_optimizer(), metrics=get_metric())
    save_model_json(mdl, params.job_id)

    return mdl


def get_dataset(is_test=False):
    """Return dataset loader."""
    loader = loaders.DatasetLoader(path=params.data_path,
                                   img_ch=3,
                                   remove_blank_images=params.remove_blank_images,
                                   remove_bad_masks=params.remove_bad_masks,
                                   resize=params.img_size,
                                   split=params.validation_split,
                                   kfold=params.kfold,
                                   kfold_nbr=params.kfold_nbr,
                                   load_test_set=is_test)

    return loader.get_dataset()


def get_generators(is_test=False):
    """Get the dataset class according to the neptune parameters."""
    datasets = get_dataset(is_test=is_test)

    # Fetch the processor
    processor = bootstrap.get_processor()

    # If len is 1, then return the test set
    if len(datasets) == 1:
        gen = data.DataGenerator(is_training=False,
                                 is_test=True,
                                 dataset=datasets[0],
                                 batch_size=20,  # For some reason, 18000/batch_size must be an integer
                                 shuffle=not is_test,
                                 processor=processor)
        return gen
    else:
        data_generators = []
        is_training = True
        for dataset in datasets:
            data_generators.append(data.DataGenerator(is_training=is_training,
                                                      is_test=False,
                                                      dataset=dataset,
                                                      batch_size=params.batch_size,
                                                      shuffle=is_training,
                                                      processor=processor))
            is_training = False

        return tuple(data_generators)


def train(initial_epoch=0):
    """Train."""
    # Frozen vs not epochs
    frozen = int(params.train_epochs * params.freeze_encoder_ratio)
    unfrozen = params.train_epochs - frozen
    train_gen, val_gen = get_generators()

    if frozen > 0:
        print("Training with frozen encoder for %d epochs" % frozen)
        mdl = get_model(freeze_encoder=True)

        mdl.summary()
        mdl.fit_generator(generator=train_gen,
                          validation_data=val_gen,
                          callbacks=get_callbacks(),
                          shuffle=True,
                          epochs=initial_epoch + frozen,
                          initial_epoch=initial_epoch)

    print("Training with no frozen encoder for %d epochs" % unfrozen)
    mdl = get_model(freeze_encoder=False)
    mdl.summary()
    mdl.fit_generator(generator=train_gen,
                      validation_data=val_gen,
                      callbacks=get_callbacks(),
                      shuffle=True,
                      epochs=initial_epoch + frozen + unfrozen,
                      initial_epoch=initial_epoch + frozen)


def ensemble(experiments):
    """Ensemble experiments."""
    import tensorflow as tf
    if not isinstance(experiments, list):
        experiments = [experiments]

    model_input = Input(shape=(101, 101, 3), name='new_input')
    model_outputs = []

    # Lambda layer, extract sigmoid only
    def get_layer(ch):
        def lambda_fcn(x):
            return K.reshape(x[:, :, :, ch], (-1, params.img_size, params.img_size, 1))
        return lambda_fcn

    for exp in experiments:
        print("Loading model from %s" % get_json_file(exp))
        with open(get_json_file(exp)) as f:
            mdl = model_from_json(f.read(), custom_objects={'tf': tf})
        mdl.load_weights(get_model_chkp(exp), by_name=True)
        mdl.name = exp
        mdl.layers.pop(0)  # Remove input layer
        logits_and_sigmoid = mdl(model_input)
        logits = Lambda(get_layer(0))(logits_and_sigmoid)

        # Set new input
        mdl = Model(model_input, logits)

        model_outputs.append(mdl.outputs[0])

    if len(model_outputs) > 1:
        y = Average()(model_outputs)
    else:
        y = model_outputs[0]

    model = Model(model_input, Activation('sigmoid')(y), name='ensemble')

    return model


def get_best_threshold(ensemble_model):
    """Run through a validation dataset, get best iou threshold."""
    train_gen, val_gen = get_generators()

    # Add metrics
    ensemble_model.compile(loss='binary_crossentropy', optimizer=get_optimizer(), metrics=get_metric(scan=True))
    preds_test = ensemble_model.evaluate_generator(generator=val_gen, verbose=True)

    # Now, simply fetch the right threshold
    ts = np.linspace(0, 1, len(preds_test))

    return ts[np.argmax(preds_test)]


def predict_on_test(experiment_id):
    """Submit."""
    mdl = ensemble(experiment_id)
    # Get threshold
    best_t = get_best_threshold(mdl)
    test_gen = get_generators(is_test=True)
    preds_test = mdl.predict_generator(generator=test_gen, verbose=True)
    return preds_test, test_gen, best_t


def submission(experiment_ids):
    """Hand submission in."""
    sub_file = get_submission_file('-'.join(experiment_ids))
    sub_dir = os.path.dirname(sub_file)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    tag = ' - '.join(experiment_ids)

    predictions, test_gen, best_t = predict_on_test(experiment_ids)
    kaggle.create_submission_file(test_gen.id, predictions, sub_file, apply_crf=False, set_threshold=best_t)
    kaggle.submit(sub_file, tag)


if __name__ == "__main__":
    # Regular training
    train()

    # Decrease learning rate by 10x
    params.set_param('learning_rate', params.learning_rate / 10)

    # Set SGD
    params.set_param('optimizer', 'SGD')
    params.set_param('freeze_encoder_ratio', 0.0)

    # Change loss
    params.set_param('loss_dice_weight', 0.0)
    params.set_param('loss_bce_weight', 0.0)
    params.set_param('loss_lovaz_weight', 1.0)
    params.set_param('focal_loss_weight', 0.0)

    # Fine-Train set epochs so that logs are correct
    params.set_param('epoch', params.train_epochs * 2)
    train(initial_epoch=params.train_epochs)

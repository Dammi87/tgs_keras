"""Methods to build the training from the Neptune params."""
import os
from keras.models import Model, Input
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


params = Params()


def get_submission_file(job_id):
    """Given experiment ID, return the model_dir."""
    model_dir = os.path.join(params.model_dir, params.model_type, job_id, 'submission.csv')
    return model_dir


def get_model_dir(job_id):
    """Given experiment ID, return the model_dir."""
    model_dir = os.path.join(params.model_dir, params.model_type, job_id, 'chkp')
    return model_dir


def get_log_dir(job_id):
    """Given experiment ID, return the log_dir."""
    log_dir = os.path.join(params.model_dir, params.model_type, job_id, 'logs')
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


def get_metric():
    """Return the metric used for training."""
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
    checkpoint = ModelCheckpoint(file_path, monitor=params.monitor_metric, verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TerminateOnNaN()]

    # Check for rest
    if params.early_stopping:
        callbacks_list.append(EarlyStopping(patience=params.early_stopping_patience, verbose=1))

    if params.reduce_lr_on_plateau:
        callbacks_list.append(ReduceLROnPlateau(factor=params.rlop_factor,
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
            return params.focal_loss_weight * loss.focal_loss(params.focal_loss_gamma, params.focal_loss_alpha)(masks, logits)

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


def get_model(experiment_id=None, freeze_encoder=True, compile_mdl=True):
    """Get the model."""
    mdl = build(params.model_type, freeze_encoder=freeze_encoder, hyper_concat=params.hyper_concat)
    mdl = maybe_resume(mdl, experiment_id)
    mdl.compile(loss=get_loss(), optimizer=get_optimizer(), metrics=get_metric())
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

    data_generators = []
    for dataset in datasets:
        data_generators.append(data.DataGenerator(is_training=not is_test,
                                                  dataset=dataset,
                                                  batch_size=params.batch_size,
                                                  shuffle=not is_test,
                                                  processor=processor))

    return tuple(data_generators)


def train():
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
                          epochs=frozen)

    print("Training with no frozen encoder for %d epochs" % unfrozen)
    mdl = get_model(freeze_encoder=False)
    mdl.summary()
    mdl.fit_generator(generator=train_gen,
                      validation_data=val_gen,
                      callbacks=get_callbacks(),
                      shuffle=True,
                      epochs=unfrozen)


def ensemble(experiments):
    """Ensemble experiments."""
    if not isinstance(experiments, list):
        experiments = [experiments]
    model_input = Input(shape=(params.img_size, params.img_size, 3), name='new_input')
    model_outputs = []

    # Lambda layer, extract sigmoid only
    def get_layer(ch):
        def lambda_fcn(x):
            return K.reshape(x[:, :, :, ch], (-1, params.img_size, params.img_size, 1))
        return lambda_fcn

    for exp in experiments:
        mdl = get_model(experiment_id=exp, compile_mdl=False)
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


def predict_on_test(experiment_id):
    """Submit."""
    mdl = ensemble(experiment_id)
    test_gen = get_generators(is_test=True)[0]
    preds_test = mdl.predict_generator(generator=test_gen, verbose=True)
    return preds_test, test_gen


def submission(experiment_ids):
    """Hand submission in."""
    sub_file = get_submission_file('-'.join(experiment_ids))
    sub_dir = os.path.dirname(sub_file)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    predictions, test_gen = predict_on_test(experiment_ids)
    kaggle.create_submission_file(test_gen.id, predictions, sub_file, apply_crf=False)
    kaggle.submit(sub_file, 'keras')


if __name__ == "__main__":
    # Regular training
    if True:
        train()

        # Decrease learning rate by 10x
        params.set_param('learning_rate', params.learning_rate / 10)

        # Set SGD
        params.set_param('optimizer', 'SGD')
        params.set_param('freeze_encoder_ratio', 0.0)

        # Change loss
        params.set_param('loss_dice_weight', 4.0)
        params.set_param('loss_bce_weight', 1.0)
        params.set_param('loss_lovaz_weight', 0.0)
        params.set_param('focal_loss_weight', 0.0)

        # Train again
        train()
    else:
        submission(['271fe092-2839-4955-98fb-fe5103285a16', '4c4f1b69-ab64-4598-86e0-0bb3c1528f96', '6df6d87e-a8fa-412f-8609-3a41da5b610b', '862f32e1-bf16-43b4-8727-61f8e2956510'])

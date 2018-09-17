"""Methods to build the training from the Neptune params."""
import os
import numpy as np
from keras.models import Model, Input, model_from_json
from keras.layers import Average, Activation, Lambda
import keras.backend as K
from solution.lib.neptune import Params
from solution.input import data, bootstrap
import solution.lib.kaggle as kaggle
from train import get_metric, get_optimizer, get_dataset


params = Params()


def get_json_file(job_id):
    """Path to json file."""
    model_dir = get_model_dir(job_id)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return os.path.join(model_dir, '%s.json' % job_id)


def get_submission_file(job_id):
    """Given experiment ID, return the model_dir."""
    model_dir = os.path.join(params.model_dir, params.encoder_decoder, job_id, 'submission.csv')
    return model_dir


def get_model_dir(job_id):
    """Given experiment ID, return the model_dir."""
    model_dir = os.path.join(params.model_dir, params.encoder_decoder, job_id, 'chkp')
    return model_dir


def get_model_chkp(job_id):
    """Given experiment ID, return the weight file."""
    file_path = os.path.join(get_model_dir(job_id), "weights.best.hdf5")
    return file_path


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
    best = ts[np.argmax(preds_test)]

    print("Found {} to be the best threshold!" % best)

    return best


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
    submission(['8b5a533e-2d60-47d4-8520-28f6850702c6', '9b237b7e-cd1a-4e9d-9b7b-c94c8581faa1'])

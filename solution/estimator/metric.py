"""Metrics to calcualte during training."""
import tensorflow as tf
import numpy as np


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    # Jiaxin fin that if all zeros, then, the background is treated as object
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0, 0.5, 1], [0, 0.5, 1]))
    intersection = temp1[0]
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        prec.append(p)

    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in, t=0.5):
    """Calculat emetric batch-wise."""
    y_pred_in = y_pred_in > t  # added by sgx 20180728
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def iou_kaggle_metric(label, pred):
    """Return kaggle metric."""
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float64)
    return metric_value


def iou_kaggle_metric_scan():
    """Same as above, except t is scanned over multiple values."""
    metric_methods = []
    for t in np.linspace(0, 1, 20):
        def outer_wrap(label, pred, t=t):
            def wrapped(label, pred, t=t):
                return iou_metric_batch(label, pred, t)
            return tf.py_func(wrapped, [label, pred], tf.float64)

        # Create a method name
        fcn_name = 'iou_%d' % (int(t * 100))
        # Assign
        exec('%s = outer_wrap' % fcn_name)
        exec('metric_methods.append(%s)' % fcn_name)

    return metric_methods

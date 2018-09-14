"""Method to perform inference, also to convert checkpoints to saved models."""
import multiprocessing
from collections import ChainMap
from solution.input.loaders import resize_img
import cv2
import pandas as pd
import os


def rle_encoding(img, order='F', format=True):
    """Run length encoding."""
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


class Consumer(multiprocessing.Process):
    """Consumer for performin a specific task."""

    def __init__(self, task_queue, result_queue):
        """Initialize consumer, it has a task and result queues."""
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        """Actual run of the consumer."""
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            # Fetch answer from task
            answer = next_task()
            self.task_queue.task_done()
            # Put into result queue
            self.result_queue.put(answer)
        return


class RleTask(object):
    """Wrap the RLE Encoder into a Task."""

    def __init__(self, idx, mask, img, apply_crf=False):
        """Save image to self."""
        self.mask = mask
        self.idx = idx
        self.img = img
        self.apply_crf = apply_crf

    def __call__(self):
        """When object is called, encode."""
        return {self.idx: rle_encoding(self.mask)}


class FastRle(object):
    """Perform RLE in paralell."""

    def __init__(self, num_consumers=2, n_items=18000):
        """Initialize class."""
        self._tasks = multiprocessing.JoinableQueue(maxsize=100)
        self._results = multiprocessing.Queue()
        self._n_consumers = num_consumers
        self._n_items = n_items

        # Initialize consumers
        self._consumers = [Consumer(self._tasks, self._results) for i in range(self._n_consumers)]
        for w in self._consumers:
            w.start()

    def add(self, idx, mask, img, apply_crf=False):
        """Add a task to perform."""
        self._tasks.put(RleTask(idx, mask, img, apply_crf))

    def get_results(self):
        """Close all tasks."""
        # Provide poison pill
        [self._tasks.put(None) for _ in range(self._n_consumers)]
        # Wait for finish
        self._tasks.join()
        # Return results
        singles = []
        while len(singles) < self._n_items:
            singles.append(self._results.get())
        return dict(ChainMap({}, *singles))


def get_latest_saved_model(model_dir):
    """Return the latest saved_model."""
    saved_models = os.path.join(model_dir, 'best_models')
    saved_chkp = sorted([int(mdl) for mdl in os.listdir(saved_models)])
    latest = saved_chkp[-1]
    path = os.path.join(saved_models, '%d' % latest)

    # Next, find the full path to the saved model
    mdl_time = os.listdir(path)

    # Return the final path
    return os.path.join(path, mdl_time[-1])


def resize_down(pred):
    """Resize with opencv."""
    return cv2.resize(pred, (101, 101), interpolation=cv2.INTER_AREA)


def create_submission_file(img_idx, img_pred, file_path, set_threshold=0.5, apply_crf=False):
    """Create submission, saved to csv file."""
    # Inference, and collect to dictionary
    n_samples = len(img_idx)
    report_every = int(float(n_samples) / 100)
    print("Found %d samples" % n_samples)
    rle = FastRle(4)
    for i, (idx, pred) in enumerate(zip(img_idx, img_pred)):
        rle.add(idx, resize_down(pred) > set_threshold, None, apply_crf)
        if i % report_every == 0:
            print("\t -> [%d/%d]" % (i, n_samples))

    result_dict = rle.get_results()
    n_results = len(result_dict)
    if n_results != n_samples:
        raise Exception('Number of results in result_dict [{}] differs from actual samples [{}]'.format(n_results, n_samples))
    sub = pd.DataFrame.from_dict(result_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(file_path)


def submit(path, msg):
    """Submit a CSV to kaggle."""
    cmd = 'kaggle competitions submit'
    cmd = '%s %s' % (cmd, 'tgs-salt-identification-challenge')
    cmd = '%s -f %s' % (cmd, path)
    cmd = '%s -m "%s"' % (cmd, msg)
    os.system(cmd)

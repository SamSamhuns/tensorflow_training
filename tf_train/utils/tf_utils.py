import os
import glob

import tensorflow as tf
from tensorflow.python.client import device_lib


def count_samples_in_tfr(data_root_path: str, ext: str = ".tfrecord") -> int:
    """
    Count number of samples in tfrecord files that are under data_root_path/**/*.tfrecord
    """
    local_device_protos = device_lib.list_local_devices()
    cpu0 = [x.name for x in local_device_protos if "cpu" in x.name.lower()][0]
    with tf.device(cpu0):
        flist = glob.glob(os.path.join(data_root_path, f"**/*{ext}"), recursive=True)
        sample_count = sum([1 for _ in tf.data.TFRecordDataset(flist)])
        return sample_count

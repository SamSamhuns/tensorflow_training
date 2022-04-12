import os
import glob

import tensorflow as tf


def count_samples_in_tfr(data_root_path: str, ext: str = ".tfrecord") -> int:
    """
    Count number of samples in tfrecord files that are under data_root_path/**/*.tfrecord
    """
    with tf.device("/cpu:0"):
        flist = glob.glob(os.path.join(data_root_path, f"**/*{ext}"), recursive=True)
        sample_count = sum([1 for _ in tf.data.TFRecordDataset(flist)])
        return sample_count

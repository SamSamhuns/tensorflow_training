import os
import json
import glob
import functools
from pathlib import Path
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.client import device_lib


def read_json(fname: str) -> dict:
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: dict, fname: str) -> None:
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def rgetattr(obj, attr, *args):
    """
    recursively get attrs. i.e. rgetattr(module, "sub1.sub2.sub3")
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


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

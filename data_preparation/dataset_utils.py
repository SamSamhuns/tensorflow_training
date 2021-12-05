import os
import sys
import math
import numpy as np
import os.path as osp
import tensorflow as tf
from tensorflow.python.client import session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(img, class_id, height, width, c):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/img': bytes_feature(img),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/channel': int64_feature(c),
        'class_id': int64_feature(class_id),
    }))


def _get_filenames_and_classes(dataset_dir):
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        path = osp.join(dataset_dir, filename)
        photo_filenames.append(path)
    return photo_filenames


def _get_tfrecord_path(shard_id, tfrecord_dir_path):
    output_path = osp.join(tfrecord_dir_path, f"{shard_id + 1:06d}.tfrecord")
    return output_path


def _convert_dataset(fnames, tfrecord_dir_path, num_shards):
    num_per_shard = int(math.ceil(len(fnames) / float(num_shards)))
    with tf.Graph().as_default():
        with session.Session():
            for shard_id in range(num_shards):
                output_tfr_path = _get_tfrecord_path(
                    shard_id, tfrecord_dir_path=tfrecord_dir_path)
                options = tf.io.TFRecordOptions()  # GZIP NONE ZLIB
                with tf.io.TFRecordWriter(output_tfr_path, options=options) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(fnames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write(
                            '\r>> Writing record %d/%d in shard %d' % (i + 1, len(fnames), shard_id + 1))
                        sys.stdout.flush()
                        data = np.load(fnames[i])
                        img = data['image']
                        class_id = data['class_id']
                        height, width, c = img.shape
                        img = img.tobytes()
                        example = image_to_tfexample(
                            img, class_id, height, width, c)
                        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

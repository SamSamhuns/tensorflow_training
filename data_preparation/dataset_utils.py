import os
import sys
import math
import glob
import os.path as osp
from typing import Union, List

import imageio
import tensorflow as tf
from tensorflow.python.client import session


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


def _get_img_to_cid_list(dataset_dir) -> List[Union[str, int]]:
    """
    Get image path to class id list
    """
    img_path_cid_list = []
    # essential to sort the glob object
    class_data_list = sorted(glob.glob(osp.join(dataset_dir, "*")))

    class_id = 0
    # for each class dir
    for dir_path in class_data_list:
        # for each img
        for img_path in glob.glob(osp.join(dir_path, "*")):
            img_path_cid_list.append([img_path, class_id])
        class_id += 1
    return img_path_cid_list


def _save_cls_map_file(dataset_dir, cls_map_path) -> None:
    """
    Save class name to class id mapping file at cls_map_path
    """
    # essential to sort the glob object
    class_data_list = sorted(glob.glob(osp.join(dataset_dir, "*")))

    class_id = 0
    with open(os.path.join(cls_map_path), 'w') as f:
        # for each class dir
        for dir_path in class_data_list:
            class_name = osp.basename(dir_path)
            f.write(str(class_id) + "\t" + class_name + "\n")
            class_id += 1


def _get_tfrecord_path(shard_id, tfrecord_dir_path) -> str:
    output_path = osp.join(tfrecord_dir_path, f"{shard_id + 1:06d}.tfrecord")
    return output_path


def _convert_dataset_to_tfr(img_path_cid_list, tfrecord_dir_path, num_shards) -> None:
    num_files = len(img_path_cid_list)
    num_per_shard = int(math.ceil(num_files / float(num_shards)))
    with tf.Graph().as_default(), session.Session():
        fail_count = 0
        for shard_id in range(num_shards):
            output_tfr_path = _get_tfrecord_path(
                shard_id, tfrecord_dir_path=tfrecord_dir_path)
            options = tf.io.TFRecordOptions()
            with tf.io.TFRecordWriter(output_tfr_path, options=options) as tfrecord_writer:
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, num_files)
                for i in range(start_ndx, end_ndx):
                    try:
                        img_path, class_id = img_path_cid_list[i]
                        img = imageio.imread(img_path, pilmode="RGB")
                        img = img[..., :3]  # drop alpha channel

                        height, width, c = img.shape
                        img = img.tobytes()
                        example = image_to_tfexample(
                            img, class_id, height, width, c)
                        tfrecord_writer.write(example.SerializeToString())
                        sys.stdout.write(
                            f'\r>> Wrote record {i+1}/{num_files} in shard {shard_id+1}')
                        sys.stdout.flush()
                    except Exception as e:
                        fail_count += 1
                        print(f"{e}. imageio could not read file {img_path}")
        if fail_count > 0:
            print(f"\n{fail_count} imgs could not read by imageio")
    sys.stdout.write('\n')
    sys.stdout.flush()

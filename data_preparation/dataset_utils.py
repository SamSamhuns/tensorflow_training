# refernce: https://keras.io/examples/keras_recipes/creating_tfrecords/
import os
import sys
import math
import glob
import os.path as osp
from typing import Union, List

import tensorflow as tf


VALID_FILE_EXTS = {'jpg', 'jpeg', 'JPEG', 'png'}


def image_feature(values):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(values).numpy()]))


def int64_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    values = values if isinstance(values, (tuple, list)) else [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    """Returns a float_list from a float / double."""
    values = values if isinstance(values, (tuple, list)) else [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=[values]))


def bytes_feature(values):
    """Returns an un-encoded bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(img, class_id, h, w, c):
    """
    Note image/{height,width,channel} are implied from image/img in the training pipeline
    """
    feature = {
        'image/img': image_feature(img),
        'image/height': int64_feature(h),
        'image/width': int64_feature(w),
        'image/channel': int64_feature(c),
        'class_id': int64_feature(class_id),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _get_img_to_cid_list(dataset_dir) -> List[Union[str, int]]:
    """
    Get image path to class id list
    """
    img_path_cid_list = []
    # essential to sort the glob object to get reproducible class ids
    class_data_list = sorted(glob.glob(osp.join(dataset_dir, "*")))

    class_id = 0
    # for each class dir
    for dir_path in class_data_list:
        if osp.isdir(dir_path):
            # img path gen with images having matching extensions
            img_gen = (path for path in glob.glob(osp.join(dir_path, "**/*"), recursive=True)
                       if osp.splitext(path)[1][1:] in VALID_FILE_EXTS)
            # for each img
            for img_path in img_gen:
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
            if osp.isdir(dir_path):
                class_name = osp.basename(dir_path)
                f.write(str(class_id) + "\t" + class_name + "\n")
                class_id += 1


def _get_tfrecord_path(shard_id, tfrecord_dir_path) -> str:
    output_path = osp.join(tfrecord_dir_path, f"{shard_id + 1:06d}.tfrecord")
    return output_path


def _convert_dataset_to_tfr(img_path_cid_list, tfrecord_dir_path, num_shards) -> None:
    num_files = len(img_path_cid_list)
    num_per_shard = int(math.ceil(num_files / float(num_shards)))
    success_count = fail_count = 0
    for shard_id in range(num_shards):
        output_tfr_path = _get_tfrecord_path(
            shard_id, tfrecord_dir_path=tfrecord_dir_path)
        with tf.io.TFRecordWriter(output_tfr_path) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, num_files)
            for i in range(start_ndx, end_ndx):
                try:
                    img_path, class_id = img_path_cid_list[i]
                    # read as RGB channels & drop alpha channel if present
                    img = tf.io.decode_image(tf.io.read_file(img_path), channels=3)

                    h, w, c = img.shape
                    example = image_to_tfexample(
                        img, class_id, h, w, c)
                    tfrecord_writer.write(example.SerializeToString())
                    sys.stdout.write(
                        f'\r>> Wrote record {i+1}/{num_files} in shard {shard_id+1}')
                    sys.stdout.flush()
                    success_count += 1
                except Exception as e:
                    fail_count += 1
                    print(f"{e}. Error reading {img_path}")
    print(f"{success_count} image(s) converted to {num_shards} tfrecords")
    if fail_count > 0:
        print(f"\n{fail_count} image(s) could not be processed into tfrecords")
    sys.stdout.write('\n')
    sys.stdout.flush()

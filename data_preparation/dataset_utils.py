# refernce: https://keras.io/examples/keras_recipes/creating_tfrecords/
import os
import glob
import tqdm
import os.path as osp
import multiprocessing
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


def _get_tfrecord_path(shard_id, num_samples_in_shard, tfrecord_dir_path) -> str:
    output_path = osp.join(
        tfrecord_dir_path, f"{shard_id+1:05d}_{num_samples_in_shard:06d}_samples.tfrecord")
    return output_path


def get_img_to_cid_list(dataset_dir) -> List[Union[str, int]]:
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


def save_cls_map_file(dataset_dir, cls_map_path) -> None:
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


def write_imgs_to_tfr_shard(img_path_cid_list, start_idx, end_idx, shard_id, output_tfr_path):
    """
    Write "image,class_id" tuples from img_path_cid_list[start_idx, end_idx)
    with shard with id shard_id to output_tfr_path tfrecord file
    """
    success_count = 0
    num_samples_in_shard = end_idx - start_idx
    with tf.io.TFRecordWriter(output_tfr_path) as tfr_writer, tqdm.tqdm(total=num_samples_in_shard) as pbar:
        for i in range(start_idx, end_idx):
            try:
                img_path, class_id = img_path_cid_list[i]
                # read as RGB channels & drop alpha channel if present
                img = tf.io.decode_image(
                    tf.io.read_file(img_path), channels=3)

                h, w, c = img.shape
                example = image_to_tfexample(
                    img, class_id, h, w, c)
                tfr_writer.write(example.SerializeToString())
                pbar.update(1)
                success_count += 1
            except Exception as e:
                print(f"{e}. Error reading {img_path}")
    return success_count


def convert_dataset_to_tfr_single_proc(img_path_cid_list, tfrecord_dir_path, num_samples_per_shard) -> None:
    num_samples = len(img_path_cid_list)

    num_shards = num_samples // num_samples_per_shard
    if num_samples % num_samples_per_shard:
        num_shards += 1  # add one record if there are any remaining samples

    success_count = 0
    for shard_id in range(num_shards):
        start_idx = shard_id * num_samples_per_shard
        end_idx = min((shard_id + 1) * num_samples_per_shard, num_samples)
        num_samples_in_shard = end_idx - start_idx if end_idx - start_idx > 0 else 0
        output_tfr_path = _get_tfrecord_path(
            shard_id, num_samples_in_shard, tfrecord_dir_path)

        success_count += write_imgs_to_tfr_shard(
            img_path_cid_list, start_idx, end_idx, shard_id, output_tfr_path)
    print(f"\n{success_count} image(s) converted to {num_shards} tfrecords")
    fail_count = num_samples - success_count
    if fail_count:
        print(f"\n{fail_count} image(s) could not be processed into tfrecords")


def convert_dataset_to_tfr_mult_proc(img_path_cid_list, tfrecord_dir_path, num_samples_per_shard) -> None:

    def _multi_process_tfr_write(func, img_path_cid_list, tfrecord_dir_path, num_shards):
        pool = multiprocessing.Pool()

        mult_func_args = []
        for shard_id in range(num_shards):
            start_idx = shard_id * num_samples_per_shard
            end_idx = min((shard_id + 1) * num_samples_per_shard, num_samples)
            num_samples_in_shard = end_idx - start_idx if end_idx - start_idx > 0 else 0
            output_tfr_path = _get_tfrecord_path(
                shard_id, num_samples_in_shard, tfrecord_dir_path)

            mult_func_args.append(
                (img_path_cid_list, start_idx, end_idx, shard_id, output_tfr_path))

        results = pool.starmap(func, mult_func_args)
        pool.close()
        pool.join()
        success_count = sum(results)
        return success_count

    num_samples = len(img_path_cid_list)
    num_shards = num_samples // num_samples_per_shard
    if num_samples % num_samples_per_shard:
        num_shards += 1  # add one record if there are any remaining samples

    success_count = _multi_process_tfr_write(
        write_imgs_to_tfr_shard, img_path_cid_list, tfrecord_dir_path, num_shards)
    print(f"\n{success_count} image(s) converted to {num_shards} tfrecords")
    fail_count = num_samples - success_count
    if fail_count:
        print("\n{fail_count} image(s) could not be processed into tfrecords")

from functools import partial
from pathlib import Path
import tensorflow as tf


@tf.function()
def _parse_train_fn(example_proto, config):
    features = tf.io.parse_single_example(
        example_proto,
        features={
            "image/img": tf.io.FixedLenFeature([], tf.string),
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "image/channel": tf.io.FixedLenFeature([], tf.int64),
            "class_id": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    image_rgb = tf.io.decode_jpeg(features["image/img"], channels=3)
    class_id = tf.cast(features["class_id"], tf.int64)

    # choose train image preprocessing
    # thresholds chosen after ensuring the image augmentations repr realistic changes
    image_rgb = tf.image.central_crop(image_rgb, 0.8)
    image_rgb = tf.image.random_flip_left_right(image_rgb)
    image_rgb = tf.image.random_brightness(image_rgb, 0.22)
    image_rgb = tf.image.random_contrast(image_rgb, 0.9, 1.3)
    image_rgb = tf.image.random_jpeg_quality(image_rgb, 75, 100)
    image_rgb = tf.image.random_saturation(image_rgb, 0.6, 1.4)
    # image_rgb = tf.cast(image_rgb, tf.float32, name='cast_input1') / 255.0
    # image_rgb = tf.keras.layers.GaussianNoise(stddev=0.005)(image_rgb, True)  # adds too much noise
    # image_rgb = tf.cast(image_rgb * 255.0, tf.uint8, name='cast_input1')
    # image_rgb = tf.image.random_flip_up_down(image_rgb) # unrealistic data alteration
    # image_rgb = tf.image.random_hue(image_rgb, 0.9) # significantly alters image, dont use
    in_h, in_w, _ = config.model.args.input_shape
    image_rgb = tf.image.resize(
        image_rgb,
        size=(in_w, in_h),
        method=tf.image.ResizeMethod.BICUBIC,
        preserve_aspect_ratio=False,
        antialias=True,
        name="Bicubic_upsampling2",
    )
    image_rgb = tf.clip_by_value(image_rgb, 0.0, 255.0, name=None)
    image_rgb = tf.cast(image_rgb, tf.float32, name="cast_input1")

    # apply preprocessing func from preloaded parent module
    image_rgb = config.model.parent_module.preprocess_input(image_rgb)
    feature_dict = {"input_img": image_rgb}
    one_hot_class = tf.one_hot(
        class_id, config["data"]["num_classes"], on_value=1, off_value=0
    )

    gt_dict = {config.model.type: one_hot_class}
    if config["optimization"]["quantize"]["during_training_quantization"]:
        gt_dict = {"quant_" + key: val for key, val in gt_dict.items()}

    if config["optimization"]["cluster"]["use_clustering"]:
        gt_dict = {"cluster_" + key: val for key, val in gt_dict.items()}

    if config["optimization"]["prune"]["use_pruning"]:
        gt_dict = {"prune_low_magnitude_" + key: val for key, val in gt_dict.items()}

    return feature_dict, gt_dict


@tf.function()
def _parse_val_fn(example_proto, config):
    features = tf.io.parse_single_example(
        example_proto,
        features={
            "image/img": tf.io.FixedLenFeature([], tf.string),
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "image/channel": tf.io.FixedLenFeature([], tf.int64),
            "class_id": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    image_rgb = tf.io.decode_jpeg(features["image/img"], channels=3)
    class_id = tf.cast(features["class_id"], tf.int64)

    image_rgb = tf.image.central_crop(image_rgb, 0.8)
    image_rgb = tf.cast(image_rgb, tf.float32, name="cast_input2")

    in_h, in_w, _ = config.model.args.input_shape
    image_rgb = tf.image.resize(
        image_rgb,
        size=(in_w, in_h),
        method=tf.image.ResizeMethod.BICUBIC,
        preserve_aspect_ratio=False,
        antialias=True,
        name="Bicubic_upsampling2",
    )
    image_rgb = tf.clip_by_value(image_rgb, 0.0, 255.0, name=None)

    image_rgb = config.model.parent_module.preprocess_input(image_rgb)
    feature_dict = {"input_img": image_rgb}
    one_hot_class = tf.one_hot(
        class_id, config["data"]["num_classes"], on_value=1, off_value=0
    )

    gt_dict = {config.model.type: one_hot_class}
    if config["optimization"]["quantize"]["post_training_quantization"]:
        gt_dict = {"quant_" + key: val for key, val in gt_dict.items()}

    if config["optimization"]["cluster"]["use_clustering"]:
        gt_dict = {"cluster_" + key: val for key, val in gt_dict.items()}

    if config["optimization"]["prune"]["use_pruning"]:
        gt_dict = {"prune_low_magnitude_" + key: val for key, val in gt_dict.items()}

    return feature_dict, gt_dict


def train_input_fn(config, bsize=None):
    tfr_parallel_read = config["data"]["tfrecord_parallel_read"]
    bsize = config["data"]["train_bsize"] if bsize is None else bsize
    train_dir = config["data"]["train_data_dir"]
    tfr_filenames = Path(train_dir).glob("*.tfrecord")
    tfr_filenames = list(map(str, tfr_filenames))
    if not tfr_filenames:
        raise ValueError(
            f'No tfrecord files found in "{train_dir}".'
            " Ensure train dir is correctly specified in json config."
        )

    dataset = tf.data.TFRecordDataset(
        filenames=tfr_filenames,
        compression_type=None,
        buffer_size=None,
        num_parallel_reads=tfr_parallel_read,
    )
    loaded_parse_train_fn = partial(_parse_train_fn, config=config)
    dataset = dataset.map(loaded_parse_train_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.cache()
    dataset = dataset.shuffle(bsize * 4, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(bsize, drop_remainder=False)
    return dataset


def val_input_fn(config, bsize=None):
    tfr_parallel_read = config["data"]["tfrecord_parallel_read"]
    bsize = config["data"]["val_bsize"] if bsize is None else bsize
    val_dir = config["data"]["val_data_dir"]
    tfr_filenames = Path(val_dir).glob("*.tfrecord")
    tfr_filenames = list(map(str, tfr_filenames))
    if not tfr_filenames:
        raise ValueError(
            f'No tfrecord files found in "{val_dir}".'
            " Ensure val dir is correctly specified in json config."
        )

    dataset = tf.data.TFRecordDataset(
        filenames=tfr_filenames,
        compression_type=None,
        buffer_size=None,
        num_parallel_reads=tfr_parallel_read,
    )
    loaded_parse_val_fn = partial(_parse_val_fn, config=config)
    dataset = dataset.map(loaded_parse_val_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(bsize * 4, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(bsize, drop_remainder=False)
    return dataset


def representative_data_gen(config):
    iter_obj = iter(train_input_fn(config, bsize=5))
    while True:
        try:
            features, _ = next(iter_obj)
            yield [features["input_img"]]
        except StopIteration:
            break

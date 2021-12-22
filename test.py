# set env ars from .env before importing any python libraries
from dotenv import load_dotenv
load_dotenv(".env")

import tqdm
import argparse
import tensorflow as tf
from tf_train.config_parser import ConfigParser


def get_class_name_list(mapping_file):
    """ mapping_file must have fmt
    0   class1
    1   class2
    2   class3
    """
    map_list = []
    with open(mapping_file, 'r') as fmap:
        for i, line in enumerate(fmap):
            line = line.strip().split('\t')
            map_list.append(line[1])
    return map_list


def test(config):
    model = tf.keras.models.load_model(config.resum)
    h, w, _ = config.model.args.input_shape
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config["data"]["test_data_dir"],
        labels='inferred',
        class_names=get_class_name_list(config["tester"]),
        image_size=(h, w),
        seed=None,
        shuffle=False,
        batch_size=32)

    # print(test_ds.class_names)
    total_classes = config["data"]["num_classes"]
    top1_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    top2_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=2)
    top3_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=3)

    for image_batch, labels_batch in tqdm.tqdm(test_ds):
        image_batch = tf.cast(image_batch, tf.float32)
        image_batch = config.model.parent_module.preprocess_input(image_batch)
        output_batch = model(image_batch)
        top1_acc.update_state(tf.one_hot(labels_batch, total_classes), output_batch)
        top2_acc.update_state(tf.one_hot(labels_batch, total_classes), output_batch)
        top3_acc.update_state(tf.one_hot(labels_batch, total_classes), output_batch)
    print(f"Statistics for model: {config.resume}")

    print(f"Top 1 accuracy: {top1_acc.result().numpy() * 100:.2f}%")
    print(f"Top 2 accuracy: {top2_acc.result().numpy() * 100:.2f}%")
    print(f"Top 3 accuracy: {top3_acc.result().numpy() * 100:.2f}%")


def main():
    args = argparse.ArgumentParser(description='Tensorflow Testing')
    args.add_argument('-c', '--config', default="config/train_image_clsf.json", type=str,
                      help="config file path (default: %(default)s)")
    args.add_argument('-r', '--resume', required=True, type=str,
                      help="path to checkpoint to use for testing")
    config = ConfigParser.from_args(args)
    test(config)


if __name__ == "__main__":
    main()

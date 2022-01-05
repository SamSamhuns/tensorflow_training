# set env ars from .env before importing any python libraries
from dotenv import load_dotenv
load_dotenv(".env")

import tqdm
import argparse
import numpy as np
import tensorflow as tf
import tf_train.metric as module_metric
from tf_train.config_parser import ConfigParser


def get_class_name_list(mapping_file):
    """ mapping_file must have fmt
    0   class1
    1   class2
    2   class3
    """
    map_list = []
    with open(str(mapping_file), 'r') as fmap:
        for i, line in enumerate(fmap):
            line = line.strip().split('\t')
            map_list.append(line[1])
    return sorted(map_list)


def test(config):
    logger = config.get_logger('train')

    model = tf.keras.models.load_model(config.resume)
    h, w, _ = config.model.args.input_shape

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config["data"]["test_data_dir"],
        labels='inferred',
        class_names=get_class_name_list(
            config["tester"]["class_map_txt_path"]),
        image_size=(h, w),
        seed=None,
        shuffle=False,
        batch_size=32)

    # print(test_ds.class_names)
    n_cls = config["data"]["num_classes"]

    agg_pred, agg_label = [], []
    for image_batch, labels_batch in tqdm.tqdm(test_ds):
        image_batch = tf.cast(image_batch, tf.float32)
        image_batch = config.model.parent_module.preprocess_input(image_batch)
        output_batch = model(image_batch)

        agg_pred.append(output_batch.numpy())
        agg_label.append(labels_batch.numpy())
    # shape=(len dataloader, bsize, n_cls)
    agg_pred = np.concatenate(agg_pred, axis=0)
    # shape=(len dataloader, bsize)
    agg_label = np.concatenate(agg_label, axis=0)
    # combine dataloader len & bsize axes
    agg_pred = agg_pred.reshape(-1, agg_pred.shape[-1])
    agg_label = agg_label.flatten()

    logger.info(f"Statistics for model: {config.resume}")
    met_val_dict = {}
    met_func_dict = {_metric: config.init_ftn(["test_metrics", _metric], module_metric, num_classes=n_cls)
                     if _metric in {"acc_per_class", "confusion_matrix"}
                     else config.init_ftn(["test_metrics", _metric], module_metric)
                     for _metric in config["test_metrics"]}
    for met, met_func in met_func_dict.items():
        met_val_dict[met] = met_func(agg_label, agg_pred)

    log = {met: met_val for met, met_val in met_val_dict.items()}
    logger.info(f"test: {(log)}")


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

# set env ars from .env before importing any python libraries
from dotenv import load_dotenv
load_dotenv(".env")

import io
import time
import argparse
from datetime import datetime
from contextlib import redirect_stdout

import tqdm
import numpy as np
import tensorflow as tf
import tf_train.metric as module_metric
from tf_train.config_parser import ConfigParser


def get_class_name_list(mapping_file: str):
    """ mapping_file must have fmt
    0   class1
    1   class2
    2   class3
    """
    map_list = []
    with open(str(mapping_file), 'r') as fmap:
        for line in fmap:
            line = line.strip().split('\t')
            map_list.append(line[1])
    return sorted(map_list)


def test(config: ConfigParser):
    config.setup_logger('test')

    model = tf.keras.models.load_model(config.resume)
    h, w, _ = config.model.args.input_shape
    classes_list = get_class_name_list(config["data"]["class_map_txt_path"])

    t0 = time.time()
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config["data"]["test_data_dir"],
        labels='inferred',
        class_names=classes_list,
        image_size=(h, w),
        seed=config["seed"],
        shuffle=False,
        batch_size=config["data"]["test_bsize"])

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

    # stream model summary to logger
    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    model_summary = f.getvalue()
    config.logger.info(model_summary)
    config.logger.info(f"Classes_list: {classes_list}")
    config.logger.info(f"Statistics for model: {config.resume}")

    met_val_dict = {}
    met_func_dict = {}
    # load metric funcs with necessary params
    for _metric in config["test_metrics"]:
        if _metric in {"acc_per_class", "confusion_matrix"} or config["test_metrics"][_metric]["type"] in {"top_k_acc"}:
            mfunc = config.init_ftn(["test_metrics", _metric], module_metric,
                                    num_classes=n_cls)
        elif _metric in {"plot_confusion_matrix"}:
            mfunc = config.init_ftn(["test_metrics", _metric], module_metric,
                                    target_names=classes_list, savepath=str(config.log_dir / "cm.jpg"))
        else:
            mfunc = config.init_ftn(["test_metrics", _metric], module_metric)
        met_func_dict[_metric] = mfunc
    # run metric funcs on agg label and pred
    for met, met_func in met_func_dict.items():
        met_val_dict[met] = met_func(agg_label, agg_pred)

    log = {met: met_val for met, met_val in met_val_dict.items()}
    config.logger.info(f"test: {(log)}")
    t1 = time.time()
    config.logger.info(f"Time taken for testing: {t1 - t0:.3f}s")


def main():
    parser = argparse.ArgumentParser(description='Tensorflow Testing')
    parser.add_argument('--cfg', '--config', type=str, dest="config", default="config/train_image_clsf.json",
                        help="config file path (default: %(default)s)")
    parser.add_argument('-r', '--resume', type=str, dest="resume", required=True,
                        help="path to checkpoint to use for testing")
    parser.add_argument('--id', '--run_id', type=str, dest="run_id", default="test_" + datetime.now().strftime(r'%Y%m%d_%H%M%S'),
                        help='unique identifier for test process. Annotates test logs. (default: %(default)s)')
    config = ConfigParser.from_args(parser, options=[])
    test(config)


if __name__ == "__main__":
    main()

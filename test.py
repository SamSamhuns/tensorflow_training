from dotenv import load_dotenv
load_dotenv(".env")
# set env ars from .env before importing any python libraries
from tf_train.config_parser import ConfigParser
import tf_train.metric as module_metric
import tensorflow as tf
import numpy as np
import tf_keras
import tqdm
from contextlib import redirect_stdout
from datetime import datetime
import argparse
import time
import os
import io


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
    # tf.keras.models.load_model is not working so tf_keras.models.load_model is used
    model = tf_keras.models.load_model(config.resume_checkpoint)

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
    config.logger.info(f"Statistics for model: {config.resume_checkpoint}")

    met_val_dict = {}
    met_func_dict = {}
    # load metric funcs with necessary params
    for _metric in config["test_metrics"]:
        if _metric in {"acc_per_class", "confusion_matrix"} or config["test_metrics"][_metric]["type"] in {"top_k_acc"}:
            mfunc = config.init_ftn(["test_metrics", _metric], module_metric,
                                    num_classes=n_cls)
        elif _metric in {"plot_confusion_matrix"}:
            mfunc = config.init_ftn(["test_metrics", _metric], module_metric,
                                    target_names=classes_list, savepath=os.path.join(config.logs_dir, "cm.jpg"))
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
    parser.add_argument(
        '--cfg', '--config', type=str, dest="config", required=True,
        help="YAML config file path.")
    parser.add_argument(
        '-r', '--resume_checkpoint', type=str, dest="resume_checkpoint", required=True,
        help="Path to checkpoint to use for testing")
    parser.add_argument(
        '--id', '--run_id', type=str, dest="run_id", default="test_" + datetime.now().strftime(r'%Y%m%d_%H%M%S'),
        help='Unique identifier for test process. Annotates test logs. (default: %(default)s)')
    parser.add_argument(
        "-o", "--override", type=str, nargs="+", dest="override", default=None,
        help="Override config params. Must match keys in YAML config. "
        "e.g. -o seed=1 dataset.type=NewDataType model.layers=[64,128,256] model.layers[2]=512 (default: %(default)s)")
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=False,
        help="Run training in verbose mode (default: %(default)s)")
    args = parser.parse_args()

    # To override key-value params from YAML file,
    # match the YAML kv structure for any additional args above
    # keys-val pairs can have nested structure separated by colons
    yaml_modification = {
        "trainer.args.resume_checkpoint": args.resume_checkpoint,
    }
    # get custom omegaconf DictConfig-like obj
    cfg = ConfigParser.from_args(args, yaml_modification)
    test(cfg)


if __name__ == "__main__":
    main()

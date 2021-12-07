from __future__ import print_function
import os
import random
import argparse
from dataset_utils import _get_filenames_and_classes, _convert_dataset

random_seed = 42  # for reproducability
CUDA_DEV = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEV
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def convert_to_tfrecord(npz_dir_path, tfrecord_dir_path, num_shards):
    npz_fnames = _get_filenames_and_classes(npz_dir_path)

    random.seed(random_seed)
    for _ in range(5):
        random.shuffle(npz_fnames)
    training_fnames = npz_fnames
    _convert_dataset(fnames=training_fnames,
                     tfrecord_dir_path=tfrecord_dir_path,
                     num_shards=num_shards)
    print(f'\nFinished converting the {tfrecord_dir_path} dataset')


def main():
    parser = argparse.ArgumentParser("""
        Convert img dataset into npz files with embedded class label info:
        eg. python _2_convert_dataset_to_npz -nd data/npz_bird_dataset/train
                                             -td data/tfrecord_bird_dataset/train
                                             -ns 100""")
    parser.add_argument('-nd', '--npz_dir_path',
                        type=str, required=True,
                        help="Source npz dir path containing npz img files with class label info")
    parser.add_argument('-td', '--tfrecord_dir_path',
                        type=str, required=True,
                        help="Dir path where tfrecord files will be stored for efficient reading")
    parser.add_argument('-ns', '--num_shards',
                        type=int, default=100,
                        help="Num of tfrecord shards. Ideally there should be approx 200 imgs per shard. (default %(default)s)")
    args = parser.parse_args()
    npz_path = args.npz_dir_path
    tfr_path = args.tfrecord_dir_path
    num_shards = args.num_shards
    os.makedirs(tfr_path, exist_ok=True)

    convert_to_tfrecord(npz_path, tfr_path, num_shards)


if __name__ == "__main__":
    main()

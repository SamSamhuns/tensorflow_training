import os
import time
import random
import argparse

random_seed = 42
random.seed(random_seed)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dataset_utils import get_img_to_cid_list, save_cls_map_file
from dataset_utils import (
    convert_dataset_to_tfr_single_proc,
    convert_dataset_to_tfr_mult_proc,
)

# ################### Source Data Organization ######################
# dataset
#       |_ class_1
#                 |_ img1
#                 |_ img2
#                 |_ ....
#       |_ class_2
#                 |_ img1
#                 |_ img2
#                 |_ ....
#       ...
# However, a recursive search is done for each class sub-dir so that
# the following structure is also valid, but the script removes any
# underlying subset partition for classes
# dataset
#       |_ class_1
#                 |_ 00
#                     |_ img1
#                     |_ img2
#                 |_ 01
#                     |_ img1
#                     |_ img2
# ###################################################################


def convert_to_tfrecord(
    img_dir_path: str,
    tfrecord_dir_path: str,
    cls_map_path: str,
    num_samples_per_shard: int,
    use_mult: bool,
):
    t0 = time.time()
    img_path_cid_list = get_img_to_cid_list(img_dir_path)
    random.shuffle(img_path_cid_list)
    save_cls_map_file(img_dir_path, cls_map_path)
    if use_mult:
        convert_dataset_to_tfr_mult_proc(
            img_path_cid_list, tfrecord_dir_path, num_samples_per_shard
        )
    else:
        convert_dataset_to_tfr_single_proc(
            img_path_cid_list, tfrecord_dir_path, num_samples_per_shard
        )
    t1 = time.time()
    print(f"With multiprocessing set to {use_mult}")
    print(
        f"Converted data in {img_dir_path} to tfrecords in {tfrecord_dir_path} in {t1 - t0:.2f}s"
    )


def main():
    parser = argparse.ArgumentParser("""
        Convert img dataset into npz files with embedded class label info:
        eg. python _2_convert_dataset_to_npz -sd data/split_bird_dataset/train
                                             -td data/tfrecord_bird_dataset/train
                                             -ns 100""")
    parser.add_argument(
        "--sd",
        "--source_dir_path",
        type=str,
        dest="source_dir_path",
        required=True,
        help="Source img dir path containing img files organized into subdirs with class names",
    )
    parser.add_argument(
        "--td",
        "--tfrecord_dir_path",
        type=str,
        dest="tfrecord_dir_path",
        required=True,
        help="Dir path where tfrecord files will be stored for efficient reading",
    )
    parser.add_argument(
        "--cp",
        "--cmap_txt_path",
        type=str,
        dest="cmap_txt_path",
        default="data/dataset_classmap.txt",
        help="""Path to txt file with info on class label and corresponding name. (default %(default)s)""",
    )
    parser.add_argument(
        "--ns",
        "--num_samples_per_shard",
        type=int,
        dest="num_samples_per_shard",
        default=100,
        help="Num of samples per tfr shards. Ideally ~100 imgs per shard. (default %(default)s)",
    )
    parser.add_argument(
        "--mt",
        "--multiprocessing",
        action="store_true",
        dest="multiprocessing",
        help="Create tfrecords with multiprocessing. WARNING: Can slow system (default: %(default)s)",
    )
    args = parser.parse_args()
    os.makedirs(args.tfrecord_dir_path, exist_ok=True)
    print(f"Class name to id mapping txt file written to {args.cmap_txt_path}")

    convert_to_tfrecord(
        args.source_dir_path,
        args.tfrecord_dir_path,
        args.cmap_txt_path,
        args.num_samples_per_shard,
        args.multiprocessing,
    )


if __name__ == "__main__":
    main()

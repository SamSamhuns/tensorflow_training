import os
import random
import argparse

random_seed = 42
random.seed(random_seed)
CUDA_DEV = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEV
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataset_utils import _get_img_to_cid_list, _save_cls_map_file, _convert_dataset_to_tfr


def convert_to_tfrecord(img_dir_path, tfrecord_dir_path, cls_map_path, num_shards):
    img_path_cid_list = _get_img_to_cid_list(img_dir_path)
    _save_cls_map_file(img_dir_path, cls_map_path)

    for _ in range(5):
        random.shuffle(img_path_cid_list)
    _convert_dataset_to_tfr(img_path_cid_list=img_path_cid_list,
                            tfrecord_dir_path=tfrecord_dir_path,
                            num_shards=num_shards)
    print(f'\nFinished converting {img_dir_path} to {tfrecord_dir_path}')


def main():
    parser = argparse.ArgumentParser("""
        Convert img dataset into npz files with embedded class label info:
        eg. python _2_convert_dataset_to_npz -sd data/split_bird_dataset/train
                                             -td data/tfrecord_bird_dataset/train
                                             -ns 100""")
    parser.add_argument('-sd', '--source_dir_path',
                        type=str, required=True,
                        help="Source img dir path containing img files organized into subdirs with class names")
    parser.add_argument('-td', '--tfrecord_dir_path',
                        type=str, required=True,
                        help="Dir path where tfrecord files will be stored for efficient reading")
    parser.add_argument('-cp', '--class_map_txt_path',
                        type=str, default="data/dataset_classmap.txt",
                        help="""Path to txt file with info on class label and corresponding name. (default %(default)s)""")
    parser.add_argument('-ns', '--num_shards',
                        type=int, default=100,
                        help="Num of tfrecord shards. Ideally there should be approx 200 imgs per shard. (default %(default)s)")
    args = parser.parse_args()
    img_path = args.source_dir_path
    tfr_path = args.tfrecord_dir_path
    cls_map_path = args.class_map_txt_path
    num_shards = args.num_shards
    os.makedirs(tfr_path, exist_ok=True)

    print(f"Class name to class id mapping txt file written to {cls_map_path}")
    convert_to_tfrecord(img_path, tfr_path, cls_map_path, num_shards)


if __name__ == "__main__":
    main()

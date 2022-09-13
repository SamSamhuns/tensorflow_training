#  splits a directory with object classes in different subdirectories into
#  train, test and optionally val sub-directory with the same class sub-dir
#  structure

import os
import glob
import shutil
import random
import argparse
from tqdm import tqdm
import os.path as osp
from typing import List


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

# #################### Data Configurations here #####################
# example source data path = "data/sample_bird_dataset"
# example target data path = "data/split_bird_dataset"
VALID_FILE_EXTS = {'jpg', 'jpeg', 'JPEG', 'png'}
random.seed(42)
# ###################################################################


def create_dir_and_copy_files(dir: str, fpath_list: List[str]) -> None:
    """dir: directory where files will be copied to
    fpath_list: list of file paths which will be copied to dir
    """
    os.makedirs(dir, exist_ok=True)
    for file in fpath_list:
        shutil.copy(file, dir)


def write_fpaths_to_file(fpath_list, txt_path, mode='w'):
    """
    Write fpaths in fpath_list to txt_path
    """
    with open(txt_path, mode) as fp:
        [fp.write(path + '\n') for path in fpath_list]


def split_train_test(source_img_dir, splitted_img_dir, val_split, test_split) -> None:
    if (val_split + test_split) > 1:
        raise ValueError(
            f"val {val_split} + test {test_split} = {val_split + test_split} split cannot exceed 1.0")
    train_dir = osp.join(splitted_img_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    train_info_txt = osp.join(splitted_img_dir, "train_paths.txt")
    write_fpaths_to_file([], train_info_txt)  # create empty train_paths.txt
    if val_split > 0:
        val_dir = osp.join(splitted_img_dir, "val")
        val_info_txt = osp.join(splitted_img_dir, "val_paths.txt")
        write_fpaths_to_file([], val_info_txt)  # create empty val_paths.txt
        os.makedirs(val_dir, exist_ok=True)
    if test_split > 0:
        test_dir = osp.join(splitted_img_dir, "test")
        test_info_txt = osp.join(splitted_img_dir, "test_paths.txt")
        os.makedirs(test_dir, exist_ok=True)
        write_fpaths_to_file([], test_info_txt)  # create empty test_paths.txt

    dir_list = glob.glob(osp.join(source_img_dir, "*"))

    # for each class in source data
    for dir_path in tqdm(dir_list):
        class_name = dir_path.split("/")[-1]  # get class name

        f_list = [file for file in glob.glob(osp.join(dir_path, "**/*"), recursive=True)
                  if osp.splitext(file)[1][1:] in VALID_FILE_EXTS]
        random.shuffle(f_list)

        val_size, test_size = 0, 0
        if val_split > 0:
            val_size = int(len(f_list) * val_split)
            val_paths = [f_list[i] for i in range(val_size)]
            class_val_dir = osp.join(val_dir, class_name)
            create_dir_and_copy_files(class_val_dir, val_paths)
            write_fpaths_to_file(val_paths, val_info_txt, mode='a')
        if test_split > 0:
            test_size = int(len(f_list) * test_split)
            test_paths = [f_list[i + val_size] for i in range(test_size)]
            class_test_dir = osp.join(test_dir, class_name)
            create_dir_and_copy_files(class_test_dir, test_paths)
            write_fpaths_to_file(test_paths, test_info_txt, mode='a')

        train_paths = [f_list[val_size + test_size + i]
                       for i in range(len(f_list) - (val_size + test_size))]
        class_train_dir = osp.join(train_dir, class_name)
        create_dir_and_copy_files(class_train_dir, train_paths)
        write_fpaths_to_file(train_paths, train_info_txt, mode='a')


def main():
    """By default dataset is spit in train-val-test in ratio 85:5:10.
    """
    parser = argparse.ArgumentParser("""
                Split dataset into train, val, and test.
                If val and test split percentages are not provided,
                default val and test splits are set to 5% and 10% resp.""")
    parser.add_argument('--sd', '--source_data_path',
                        type=str, dest="source_data_path", required=True,
                        help="source dataset path with class imgs inside folders""")
    parser.add_argument('--td', '--target_data_path',
                        type=str, dest="target_data_path", required=True,
                        help="Target dataset path where imgs will be sep into train, val or test")
    parser.add_argument('--vs', '--val_split',
                        type=float, dest="val_split", default=0.,
                        help='Val data split proportion. Pass 0.05 for 5% split and so on. (default: %(default)s)')
    parser.add_argument('--ts', '--test_split',
                        type=float, dest="test_split", default=0.,
                        help='Test data split proportion. Pass 0.05 for 5% split and so on. (default: %(default)s)')
    args = parser.parse_args()
    split_train_test(
        args.source_data_path, args.target_data_path, args.val_split, args.test_split)


if __name__ == "__main__":
    main()

import os
import glob
import shutil
import argparse
from tqdm import tqdm
import os.path as osp

VALID_FILE_EXTS = {"jpg", "jpeg", "JPEG", "png"}

# #################### Data Organization ############################
#   source_data
#          |_ dataset
#                   |_ class_1
#                             |_ img1
#                             |_ img2
#                             |_ ....
#                   |_ class_2
#                             |_ img1
#                             |_ img2
#                             |_ ....
#                   ...
# ###################################################################


def safe_copy(file_path, out_dir, dst=None):
    """Safely copy a file to the specified directory. If a file with the same name already
    exists, the copied file name is altered to preserve both.

    :param str file_path: Path to the file to copy.
    :param str out_dir: Directory to copy the file into.
    :param str dst: New name for the copied file. If None, use the name of the original
        file.
    """
    name = dst or osp.basename(file_path)
    if not osp.exists(osp.join(out_dir, name)):
        shutil.copy(file_path, osp.join(out_dir, name))
    else:
        base, extension = osp.splitext(name)
        i = 1
        while osp.exists(osp.join(out_dir, f"{base}_{i}{extension}")):
            i += 1
        shutil.copy(file_path, osp.join(out_dir, f"{base}_{i}{extension}"))


def duplicate_data_dir(source_img_dir, duplicated_img_dir, target_number):
    target_dir = duplicated_img_dir
    os.makedirs(target_dir, exist_ok=True)

    dir_list = glob.glob(osp.join(source_img_dir, "*"))

    # for each class in source data
    for dir_path in tqdm(dir_list):
        class_name = dir_path.split("/")[-1]  # get class name
        f_list = [
            file
            for file in sorted(glob.glob(osp.join(dir_path, "*")))
            if osp.splitext(file)[1][1:] in VALID_FILE_EXTS
        ]

        class_target_dir = osp.join(target_dir, class_name)
        if osp.exists(class_target_dir):
            print(f"Skipping {class_target_dir}")
            continue
        os.makedirs(class_target_dir, exist_ok=True)
        f_count = 0
        while f_count <= target_number:
            for file in f_list:
                f_count += 1
                if f_count > target_number:
                    break
                safe_copy(file, class_target_dir)
            if f_count == 0:  # no files in dir
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sd",
        "--source_data_path",
        type=str,
        dest="source_data_path",
        required=True,
        help="Path with class imgs inside subdirs",
    )
    parser.add_argument(
        "--td",
        "--target_data_path",
        type=str,
        dest="target_data_path",
        required=True,
        help="Path where imgs will be saved in subdirs repr classes with number matching target_number",
    )
    parser.add_argument(
        "-n",
        "--target_number",
        type=int,
        dest="target_number",
        default=1000,
        help="""Target size to reach for each class after duplication.
                        If class has more imgs than target_number, only target_number imgs are copied.
                        (default : %(default)s)""",
    )
    args = parser.parse_args()
    duplicate_data_dir(args.source_data_path, args.target_data_path, args.target_number)


if __name__ == "__main__":
    main()

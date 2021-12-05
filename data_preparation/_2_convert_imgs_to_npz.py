import os
import glob
import imageio
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from collections import defaultdict


def generate_npz_files(raw_data_path, npz_data_path, class_map_txt_path):
    os.makedirs(npz_data_path, exist_ok=True)
    class_id = 0
    npz_name = 1
    dir_list = glob.glob(osp.join(raw_data_path, "*"))
    class_count_dict = defaultdict(int)

    with open(os.path.join(class_map_txt_path), 'w') as f:
        for dir_path in tqdm(dir_list):  # iterate through class dirs
            split_string = dir_path.split('/')
            class_name = split_string[-1]
            f.write(str(class_id) + "\t" + class_name + "\n")
            img_list = glob.glob(osp.join(dir_path, "*"))
            for img_name in img_list:    # iterate through images foreach class
                try:
                    img = imageio.imread(img_name, pilmode="RGB")
                    img = img[..., :3]  # drop alpha channel
                    np.savez(osp.join(npz_data_path, str(npz_name).zfill(6)),
                             image=img, class_id=class_id)
                    npz_name += 1
                    class_count_dict[class_name] += 1
                except Exception as e:
                    print(f"{e}. imageio could not read file {img_name}")
            class_id += 1
    [print(cname, ':', ccount) for cname, ccount in class_count_dict.items()]


def main():
    parser = argparse.ArgumentParser("""
        Convert img dataset into npz files with embedded class label info:
        eg. python _2_convert_dataset_to_npz -rd data/split_bird_dataset/train
                                             -nd data/npz_bird_dataset/train
                                             -cp data/bird_dataset_classmap.txt""")
    parser.add_argument('-rd', '--raw_data_path',
                        type=str, required=True,
                        help="""Raw dataset path with
                        class imgs inside folders""")
    parser.add_argument('-nd', '--npz_data_path',
                        type=str, required=True,
                        help="""npz dataset path where
                        imgs will be stored as npz files with class label info""")
    parser.add_argument('-cp', '--class_map_txt_path',
                        type=str, default="data/dataset_classmap.txt",
                        help="""Path to txt file with info on class label and corresponding name. (default %(default)s)""")
    args = parser.parse_args()
    generate_npz_files(args.raw_data_path, args.npz_data_path,
                       args.class_map_txt_path)


if __name__ == "__main__":
    main()

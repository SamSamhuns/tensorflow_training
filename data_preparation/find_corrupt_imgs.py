import os
import glob
import imageio
import argparse
import os.path as osp
from tqdm import tqdm


def validate_imgs(raw_data_path, corrupt_flist_txt, remove):
    dir_list = glob.glob(osp.join(raw_data_path, "*"))
    corrupt_file_count = 0
    with open(corrupt_flist_txt, 'w') as fw:
        for dir_path in tqdm(dir_list):
            img_list = glob.glob(osp.join(dir_path, "*"))
            for img_name in img_list:
                try:
                    img = imageio.imread(img_name, pilmode="RGB")
                    img = img[..., :3]
                except Exception as e:
                    print(f"{e}. imageio could not read file {img_name}")
                    fw.write(img_name + '\n')
                    corrupt_file_count += 1
                    if os.path.exists(img_name) and remove:
                        print(f"Removing {img_name}")
                        os.remove(img_name)
    print(f"Number of corrupt images discovered: {corrupt_file_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--raw_data_path',
                        type=str, required=True,
                        help='Raw dataset path with class imgs inside folders')
    parser.add_argument('-r', '--remove',
                        action="store_true", required=False,
                        help='Remove corrupt imgs. By default, the imgs are only listed and their paths saved in a file')
    parser.add_argument('-t', '--corrupt_file_list_txt_path',
                        type=str, default="corrupt_imgs.txt",
                        help='Source dataset path with class imgs inside folders. (default %(default)s)')
    args = parser.parse_args()
    if args.remove:
        confirm_removal = input(
            "Corrupt files will be removed: Continue (Y/N)?")
        args.remove = True if confirm_removal in {
            'Y', 'y', 'yes', 'Yes'} else False

    if args.remove:
        print("Corrupt files will be removed")
    else:
        print("Corrupt files will NOT be removed")

    validate_imgs(args.raw_data_path,
                  args.corrupt_file_list_txt_path,
                  args.remove)


if __name__ == "__main__":
    main()

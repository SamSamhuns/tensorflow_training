import os
import time
import glob
import logging
import argparse
import traceback
import os.path as osp
import multiprocessing
from datetime import datetime

import av
import cv2
import tqdm
import numpy as np

VALID_FILE_EXTS = {'mp4', 'avi'}

today = datetime.today()
year, month, day, hour, minute, sec = today.year, today.month, today.day, today.hour, today.minute, today.second

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=f'logs/extraction_statistics_{year}{month}{day}_{hour}:{minute}:{sec}.log',
                    level=logging.INFO)


class FrameExtractor:
    def __init__(self, source_path, target_path, cmap_path, reshape_size, max_n_frame, pad_zeros):
        self.source_path = source_path
        self.target_path = target_path
        self.cmap_path = cmap_path
        self.reshape_size = reshape_size
        self.max_n_frame = max_n_frame
        self.pad_zeros = pad_zeros
        self.cname_to_label_dict = self.get_cname_to_label_dict(
            self.source_path)
        os.makedirs(self.target_path, exist_ok=True)
        self.save_cname_to_label_dict(self.cname_to_label_dict, self.cmap_path)

    def get_cname_to_label_dict(self, source_path):
        """
        Get class name to label dict where source_path contains class dirs with data items
        """
        # essential to sort the glob object
        class_data_list = sorted(glob.glob(osp.join(source_path, "*")))

        cname_to_label_dict = {}
        class_id = 0
        # for each class dir
        for dir_path in class_data_list:
            if osp.isdir(dir_path):
                class_name = osp.basename(dir_path)
                cname_to_label_dict[class_name] = class_id
                class_id += 1
        return cname_to_label_dict

    def save_cname_to_label_dict(self, cdict, cmap_path):
        """
        Saves the class name to label dict as class label to name mapping in a txt file at cmap_path
        """
        with open(os.path.join(cmap_path), 'w') as f:
            # for each class
            for class_name, class_id in cdict.items():
                f.write(str(class_id) + "\t" + class_name + "\n")

    def extract_img_np_arr_first_nframe(self, video_path):

        cap = av.open(video_path)
        cap.streams.video[0].thread_type = 'AUTO'
        fps = int(round(cap.streams.video[0].average_rate))
        # nframes = np.floor(cap.streams.video[0].frames)
        img_list = []
        i = 0
        save_frames_num = 0
        for frame in cap.decode(video=0):
            i += 1
            if i % fps == 0 or i == 1:
                img = np.array(frame.to_image())
                save_frames_num += 1
                if save_frames_num > self.max_n_frame:
                    break
                if self.reshape_size:
                    img = cv2.resize(img, self.reshape_size).astype(np.float32)
                img_list.append(img)
        cap.close()
        del cap
        return np.array(img_list)

    def extract_img_np_arr_uniform_nframe(self, video_path):

        cap = av.open(video_path)
        cap.streams.video[0].thread_type = 'AUTO'
        # fps = int(round(cap.streams.video[0].average_rate))
        video_nframes = np.floor(cap.streams.video[0].frames)
        # uniform sample max_n_frame frames from video
        uniform_frame_set = set(
            np.arange(0, video_nframes, video_nframes // self.max_n_frame))
        img_list = []
        i = 0
        save_frames_num = 0
        for frame in cap.decode(video=0):
            if i in uniform_frame_set:
                img = np.array(frame.to_image())
                save_frames_num += 1
                if save_frames_num > self.max_n_frame:
                    break
                if self.reshape_size:
                    img = cv2.resize(img, self.reshape_size).astype(np.float32)
                img_list.append(img)
            i += 1
        cap.close()
        del cap
        return np.array(img_list)

    def extract_and_save_img_np_arr(self, video_path, npy_path):
        try:
            np_arr = self.extract_img_np_arr_first_nframe(video_path)
            # np_arr = extract_img_np_arr_uniform_nframe(video_path)
            # if pad_zeros is set to True and video has less frames than max_n_frame
            if self.pad_zeros and len(np_arr) < self.max_n_frame:
                diff = self.max_n_frame - len(np_arr)
                if self.reshape_size:
                    h, w = self.reshape_size
                else:
                    h, w = np_arr.shape[1:3]
                np_arr = np.concatenate(
                    [np_arr, np.zeros([diff, h, w, 3])], axis=0)
            cname = video_path.split('/')[-2]
            label = self.cname_to_label_dict[cname]
            np.savez_compressed(file=npy_path, arr=np_arr, label=label)
        except Exception as e:
            print(e)
            traceback.print_exc()
            return 0
        return 1

    def extract_frames_from_video_single_process(self):
        print("Single Process Extraction")
        init_tm = time.time()
        dir_path_list = glob.glob(os.path.join(self.source_path, "*"))

        total_media_ext = 0
        # for each class in raw data
        for dir_path in tqdm.tqdm(dir_path_list):
            if not os.path.isdir(dir_path):       # skip if path is not a dir
                continue
            class_name = osp.basename(dir_path)   # get class name
            print(f"Frames will be extracted from class {class_name}")
            media_path_list = [mpath for mpath in glob.glob(osp.join(dir_path, "*"))
                               if osp.splitext(mpath)[1][1:] in VALID_FILE_EXTS]

            target_save_dir = osp.join(self.target_path, class_name)
            os.makedirs(target_save_dir, exist_ok=True)
            class_media_ext = 0
            for media_path in tqdm.tqdm(media_path_list):
                try:
                    npy_name = osp.basename(media_path).split('.')[0] + ".npy"
                    npy_frames_save_path = osp.join(target_save_dir, npy_name)

                    if osp.exists(npy_frames_save_path):  # skip pre-extracted faces
                        print(
                            f"Skipping {npy_frames_save_path} as it already exists.")
                        continue

                    class_media_ext += self.extract_and_save_img_np_arr(
                        media_path, npy_frames_save_path)
                except Exception as e:
                    print(f"{e}. Extraction failed for media {media_path}")
                    traceback.print_exc()
            total_media_ext += class_media_ext
            logging.info(
                f"{class_media_ext} frame arrays extracted for class {class_name}")
        logging.info(
            f"{total_media_ext} frame arrays extracted from {self.source_path} and saved in {self.target_path}")
        logging.info(
            f"Total time taken: {time.time() - init_tm:.2f}s")

    def extract_frames_from_video_multi_process(self):
        print("Multi Process Extraction")

        def _multi_process_np_arr_extraction(source_dir, target_dir):
            pool = multiprocessing.Pool()
            mult_func_args = []

            dir_path_list = glob.glob(os.path.join(self.source_path, "*"))
            for dir_path in tqdm.tqdm(dir_path_list):
                if not os.path.isdir(dir_path):       # skip if path is not a dir
                    continue
                class_name = osp.basename(dir_path)   # get class name
                media_path_list = [mpath for mpath in glob.glob(osp.join(dir_path, "*"))
                                   if osp.splitext(mpath)[1][1:] in VALID_FILE_EXTS]

                target_save_dir = osp.join(self.target_path, class_name)
                os.makedirs(target_save_dir, exist_ok=True)
                for media_path in media_path_list:
                    npy_name = osp.basename(media_path).split('.')[0] + ".npy"
                    npy_frames_save_path = osp.join(target_save_dir, npy_name)

                    if osp.exists(npy_frames_save_path):  # skip pre-extracted faces
                        print(
                            f"Skipping {npy_frames_save_path} as it already exists.")
                        continue
                    mult_func_args.append((media_path, npy_frames_save_path))

            results = pool.starmap(
                self.extract_and_save_img_np_arr, mult_func_args)
            pool.close()
            pool.join()
            return results

        init_tm = time.time()
        total_media_ext = _multi_process_np_arr_extraction(
            self.source_path, self.target_path)

        logging.info(
            f"{total_media_ext} frame arrays extracted from {self.source_path} and saved in {self.target_path}")
        logging.info(
            f"Total time taken: {time.time() - init_tm:.2f}s")


def main():
    """
    Data must be in the following format
    data
        |_class1
                |_ video1
                |_ video2
                ...
        |_class2
                |_ video1
                |_ video2
                ...
    """
    parser = argparse.ArgumentParser("Extract frames from a video dataset.")
    parser.add_argument('-sd', '--source_data_path',
                        type=str, required=True,
                        help="Source dataset path with class imgs inside folders")
    parser.add_argument('-td', '--target_data_path',
                        type=str, default="extracted_data",
                        help="Target dataset path where video frames will be extracted to. (default: %(default)s)")
    parser.add_argument('-mf', '--max_n_frame',
                        type=int, default=15,
                        help='Max number of frames to extract from video. (default: %(default)s)')
    parser.add_argument("-rs", "--reshape_size",
                        nargs=2, default=None,
                        help='Video frames are resized to this (w,h) -rs 224 224. (default: %(default)s)')
    parser.add_argument("-mt", "--multiprocessing",
                        action="store_true",
                        help='Extract videos with multiprocessing. WARNING: Can slow down system (default: %(default)s)')
    parser.add_argument("-pz", "--pad_zeros",
                        action="store_true",
                        help='Pad missing frames with np.zero if video has less frames than max_n_frame (default: %(default)s)')
    parser.add_argument('-cm', '--class_map_txt_path',
                        type=str, default="{TARGET_DATA_PATH}/dataset_classmap.txt",
                        help='Path to txt file where class label & name will be saved to. (default: %(default)s)')
    args = parser.parse_args()
    if args.reshape_size:  # ensure reshape_size is of type int
        args.reshape_size = tuple(map(int, args.reshape_size))

    source, target, cmap = args.source_data_path, args.target_data_path, args.class_map_txt_path
    cmap = os.path.join(args.target_data_path,
                        "dataset_classmap.txt") if cmap == "{TARGET_DATA_PATH}/dataset_classmap.txt" else cmap

    frame_ext = FrameExtractor(
        source, target, cmap, args.reshape_size, args.max_n_frame, args.pad_zeros)
    if args.multiprocessing:
        frame_ext.extract_frames_from_video_multi_process()
    else:
        frame_ext.extract_frames_from_video_single_process()


if __name__ == "__main__":
    main()

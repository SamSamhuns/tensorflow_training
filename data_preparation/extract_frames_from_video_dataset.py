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

CLASS_NAME_TO_LABEL_DICT = {"class1": 0, "class2": 1}


def extract_img_np_arr(video_path, MAX_N_FRAME, reshape_size):

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
            if save_frames_num > MAX_N_FRAME:
                break
            img = cv2.resize(img, reshape_size).astype(np.float32)
            img_list.append(img)
    cap.close()
    del cap
    return np.array(img_list)


def extract_and_save_img_np_arr(video_path, npy_path, MAX_N_FRAME, reshape_size):
    try:
        np_arr = extract_img_np_arr(video_path, MAX_N_FRAME, reshape_size)
        if len(np_arr) < MAX_N_FRAME:
            diff = MAX_N_FRAME - len(np_arr)
            h, w = reshape_size
            np_arr = np.concatenate(
                [np_arr, np.zeros([diff, h, w, 3])], axis=0)
        cname = video_path.split('/')[-2]
        label = CLASS_NAME_TO_LABEL_DICT[cname]
        np.savez_compressed(file=npy_path, image=np_arr, label=label)
    except Exception as e:
        print(e)
        return 0
    return 1


def extract_frames_from_video_single_process(source_data_path, target_data_path, reshape_size, MAX_N_FRAME):
    print("Single Process Extraction")
    init_tm = time.time()
    dir_path_list = glob.glob(os.path.join(source_data_path, "*"))

    total_media_ext = 0
    # for each class in raw data
    for dir_path in tqdm.tqdm(dir_path_list):
        if not os.path.isdir(dir_path):       # skip if path is not a dir
            continue
        class_name = osp.basename(dir_path)   # get class name
        print(f"Frames will be extracted from class {class_name}")
        media_path_list = [mpath for mpath in glob.glob(osp.join(dir_path, "*"))
                           if osp.splitext(mpath)[1][1:] in VALID_FILE_EXTS]

        target_save_dir = osp.join(target_data_path, class_name)
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

                class_media_ext += extract_and_save_img_np_arr(
                    media_path, npy_frames_save_path, MAX_N_FRAME, reshape_size)
            except Exception as e:
                print(f"{e}. Extraction failed for media {media_path}")
                traceback.print_exc()
        total_media_ext += class_media_ext
        logging.info(
            f"{class_media_ext} frame arrays extracted for class {class_name}")
    logging.info(
        f"{total_media_ext} frame arrays extracted from {source_data_path} and saved in {target_data_path}")
    logging.info(
        f"Total time taken: {time.time() - init_tm:.2f}s")


def extract_frames_from_video_multi_process(source_data_path, target_data_path, reshape_size, MAX_N_FRAME):
    print("Multi Process Extraction")

    def _multi_process_np_arr_extraction(source_dir, target_dir, reshape_size, MAX_N_FRAME):
        pool = multiprocessing.Pool()
        mult_func_args = []

        dir_path_list = glob.glob(os.path.join(source_data_path, "*"))
        for dir_path in tqdm.tqdm(dir_path_list):
            if not os.path.isdir(dir_path):       # skip if path is not a dir
                continue
            class_name = osp.basename(dir_path)   # get class name
            media_path_list = [mpath for mpath in glob.glob(osp.join(dir_path, "*"))
                               if osp.splitext(mpath)[1][1:] in VALID_FILE_EXTS]

            target_save_dir = osp.join(target_data_path, class_name)
            os.makedirs(target_save_dir, exist_ok=True)
            for media_path in media_path_list:
                npy_name = osp.basename(media_path).split('.')[0] + ".npy"
                npy_frames_save_path = osp.join(target_save_dir, npy_name)

                if osp.exists(npy_frames_save_path):  # skip pre-extracted faces
                    print(
                        f"Skipping {npy_frames_save_path} as it already exists.")
                    continue
                mult_func_args.append(
                    (media_path, npy_frames_save_path, MAX_N_FRAME, reshape_size))

        results = pool.starmap(extract_and_save_img_np_arr, mult_func_args)
        pool.close()
        pool.join()
        return results

    init_tm = time.time()
    total_media_ext = _multi_process_np_arr_extraction(
        source_data_path, target_data_path, reshape_size, MAX_N_FRAME)

    logging.info(
        f"{total_media_ext} frame arrays extracted from {source_data_path} and saved in {target_data_path}")
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
    parser.add_argument('-mf', '--MAX_N_FRAME',
                        type=int, default=15,
                        help='Max number of frames to extract from video. (default: %(default)s)')
    parser.add_argument("-rs", "--reshape_size",
                        nargs=2, default=(224, 224),
                        help='Video frames are resized to this (w,h) -rs 224 224. (default: %(default)s)')
    parser.add_argument("-mt", "--multiprocessing",
                        action="store_true",
                        help='Extract videos with multiprocessing. WARNING: Can slow down system (default: %(default)s)')
    args = parser.parse_args()
    if args.multiprocessing:
        extract_frames_from_video_multi_process(
            args.source_data_path, args.target_data_path, args.reshape_size, args.MAX_N_FRAME)
    else:
        extract_frames_from_video_single_process(
            args.source_data_path, args.target_data_path, args.reshape_size, args.MAX_N_FRAME)


if __name__ == "__main__":
    main()

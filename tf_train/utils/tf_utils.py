import os
import glob
import tqdm
import tensorflow as tf


def count_samples_in_tfr(data_root_path: str, ext: str = ".tfrecord") -> int:
    """
    Count number of samples in tfrecord files that are under data_root_path/**/*.tfrecord
    """
    print(f"Counting number of samples in tfrecord files under {data_root_path}")
    with tf.device("/cpu:0"):
        flist = glob.glob(os.path.join(data_root_path, f"**/*{ext}"), recursive=True)
        sample_count = 0
        chunk_sz = 50
        # for counting lot of tfrecord files, divide and count using chunks of tfr files
        with tqdm.tqdm(total=len(flist)) as pbar:
            for i in range(0, len(flist), chunk_sz):
                start, end = i, min(i + chunk_sz, len(flist))
                for _ in tf.data.TFRecordDataset(flist[start:end]):
                    sample_count += 1
                pbar.update(end - start)
        return sample_count

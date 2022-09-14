# TensorFlow Model Training

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/74a038f3ead74de6916808b353a34268)](https://www.codacy.com/gh/SamSamhuns/tensorflow_training/dashboard?utm_source=github.com&utm_medium=referral&utm_content=SamSamhuns/tensorflow_training&utm_campaign=Badge_Grade)[![Python 3.6](https://img.shields.io/badge/python-3.7-green.svg)](https://www.python.org/downloads/release/python-370/)[![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)](https://www.python.org/downloads/release/python-380/)[![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)](https://www.python.org/downloads/release/python-390/)

Train TensorFlow models for image/video/features classification or other tasks. Currently the repository is set to train on image classification by default.

## Requirements

Framework tested with python 3.8.x

### Install requirements with `venv`:

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Note:** When using a python `venv`, the cuda libraries must be present in the current path.

Assuming cuda was installed to `usr/local/cuda`, the `LD_LIBRARY_PATH` variable should be set to `/usr/local/cuda/lib64` and the `/usr/local/cuda/bin` must be added to the `$PATH` variable as well in the current shell source file.

```shell
# if cuda was installed to usr/local/cuda
# Set cuda LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
# add cuda bin dir to path
export PATH="$PATH:/usr/local/cuda/bin"
```

The environment variable `XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda` must also be set to the cuda directory in the `.env` file.

### Or, Install requirements with `conda`:

```shell
conda create --name tf_gpu tensorflow-gpu python=3.8 -y
conda activate tf_gpu
while read requirement; do conda install --yes $requirement; done < requirements.txt
```

Note: `Conda` sets the `cuda`, `cudnn` and `cudatoolkit` automatically, downloading non-python dependencies as well.

## Data Preparation

Assuming the data directory must be organized according to the following structure, with sub-directories having class names containing images.

i.e.

     data
        |_ src_dataset
                     |_ class_1
                               |_ img1
                               |_ img2
                               |_ ....
                     |_ class_2
                               |_ img1
                               |_ img2
                               |_ ....
                     ...

Note: ImageNet style ordering of data is also supported i.e. images ordered under subdirectories inside the class directories.

i.e.

    data
       |_ src_dataset
                   |_ class_1
                            |_ 00d
                                |_ img1
                                |_ img2
                            |_ 01
                                |_ img1
                                |_ img2
                            |_ ...
                   |_ ...

### Data Duplication and Cleaning

If all the classes do not have equal number of training samples, data Duplication can be done.

```shell
python data_preparation/duplicate_data.py --sd data/src_dataset --td data/duplicated_dataset -n NUM_TO_DUPLICATE
# find corrupt images (i.e. that cannot be opened with tf.io.decode_image)
python data_preparation/find_corrupt_imgs.py --rd data/src_dataset
```

### Train-Val-Test Splitting

Set validation and test split in fractions (i.e. 0.1). Both splits are optional.

```shell
python data_preparation/create_train_val_test_split.py --sd data/duplicated_dataset --td data/split_dataset --vs VAL_SPLIT --ts TEST_SPLIT
# to check the number of images in train, val and test dirs
bash scripts/count_files_per_subdir.sh data/split_dataset
```

### Convert Data to tfrecords for faster training

Note: The test split should not be converted into tfrecords and the original `data->class_sub_directory` format should be used.

```shell
# convert train files into train tfrecord, select NUM_SHARDS so that each shard has a size of 100 MB+
python data_preparation/convert_imgs_to_tfrecord.py --sd data/split_dataset/train --td data/tfrecord_dataset/train --cp CLASS_MAP_TXT_PATH --ns NUM_SAMPLES_PER_SHARDS
# convert val files into val tfrecord, select NUM_SHARDS so that each shard has a size of 100 MB+
python data_preparation/convert_imgs_to_tfrecord.py --sd data/split_dataset/val --td data/tfrecord_dataset/val --cp CLASS_MAP_TXT_PATH --ns NUM_SAMPLES_PER_SHARDS
# to use multiprocessing use the --mt flag
```

Note: test dataset is not converted to tfrecord as fast-loading is not a priority as we only run through the test data once.

### OPTIONAL: Video Frame Extraction

To extract frames from videos into `npy.npz` files install opencv and pyav, then run:

```shell
python data_preparation/extract_frames_from_video_dataset.py --sd SOURCE_DATA_DIR
# use -h for help
```

## Selecting model, data paths and model hyper-parameters

Configure all values in the `JSON` files inside the `config` dir. A sample config file is provided for training on the `src_dataset` directory in `config/train_image_clsf.json`.

The model information repository is located at `tf_train/model/models_info.py`. New models can be added or model parameters can be modified through this file.

### Set environment variables

Set number of GPUs to use, Tensorflow, and other system environment variables in `.env`.

## Model Training

```shell
python train.py --cfg CONFIG_JSON_PATH [-r RESUME_CHECKPOINT_PATH]
```

Notes:

-   Using the `-r` option while training will override the `resume_checkpoint` param in config json if this param is not null.
-   To add tensorflow logs to train/test logs, set `"disable_existing_loggers"` parameter to `true` in `tf_train/logging/logger_config.json`.
-   Out of Memory errors during training could be caused by large batch sizes, model size or dataset.cache() call in train preprocessing in `tf_train/pipelines/data_pipeline.py`.
-   When using mixed_float16 precision, the dtypes of the final dense and activation layers must be set to `float32`.
-   An error like: `ValueError: Unexpected result of train_function (Empty logs)` could be caused by incorrect paths to train and validation directories in the config.json files


### Training Tracking with TensorBoard

```shell
tensorboard --logdir=checkpoints/tf_logs/ --port=PORT_NUM
```

## Model Testing

Make sure to set the correct `test_data_dir` under `data` and the `class_map_txt_path` under `tester` in the json config file.
The class_map_txt_path file is generated by the `convert_imgs_to_tfrecord.py` script when converting images to tfrecord format.

```shell
python test.py --cfg CONFIG_JSON_PATH -r TEST_CHECKPOINT_PATH
```

## For Developers

Unit and integration testing with pytest

```shell
python -m pytest tf_train  # from the top project directory
```

## Docker

Set up docker to run with NVIDIA first.

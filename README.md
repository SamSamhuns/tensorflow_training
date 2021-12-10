# TensorFlow Model Training

Train TensorFlow models for image/video/features classification or other tasks. Currently the repository is set to train on image classification by default.

## Requirements

Framework tested with python 3.7.x

Install requirements inside a virtualenv or a conda env.

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Data Preparation

Assuming our dataset is organized similarly to the sample birds dataset in the `data/sample_bird_dataset` directory, with class sub-directories containing images.

### Data Duplication

```shell
$ python data_preparation/_0_data_duplication.py -rd data/sample_bird_dataset -td data/duplicated_bird_dataset -n NUM_TO_DUPLICATE
```

### Train-Val-Test Splitting

```shell
$ python data_preparation/_1_create_train_val_test_split.py -rd data/duplicated_bird_dataset -td data/split_bird_dataset -vs VAL_SPLIT -ts TEST_SPLIT
# to check the number of images in train, val and test dirs
$ bash data/count_files_per_subdir.sh data/split_bird_dataset
```

### Convert Data to tfrecords

```shell
# convert train npz files into train tfrecord, select NUM_SHARDS so that each shard has a size of 100 MB+
$ python data_preparation/_2_convert_imgs_to_tfrecord.py -id data/split_bird_dataset/train -td data/tfrecord_bird_dataset/train -cp CLASS_MAP_TXT_PATH -ns NUM_SHARDS
# convert val npz files into val tfrecord, select NUM_SHARDS so that each shard has a size of 100 MB+
$ python data_preparation/_2_convert_imgs_to_tfrecord.py -id data/split_bird_dataset/val -td data/tfrecord_bird_dataset/val -cp CLASS_MAP_TXT_PATH -ns NUM_SHARDS
```

Note: test dataset is not converted to tfrecord as fast-loading is not a priority as we only run through the test data once.

## Model Selection

Select model in `_1_std_headers.py`

## Setting data paths and model hyper-parameters

Set training hyperparameters in `_1_std_headers.py`

## Model Training

```shell
$ python _9_training.py
```

### Training Tracking with TensorBoard

```shell
$ tensorboard --logdir=logs/ --port=PORT_NUM
```

## Model Testing

```shell
$ python _10_testing.py
```

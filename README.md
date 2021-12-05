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
$ python data_preparation/_1_create_train_test_split.py -rd data/duplicated_bird_dataset -td data/split_bird_dataset -vs VAL_SPLIT -ts TEST_SPLIT
# to check the number of images in train, val and test dirs
$ bash data/count_files_per_subdir.sh data/split_bird_dataset
```

### Convert Data to npz

```shell
# convert train data to npz
$ python data_preparation/_2_convert_dataset_to_npz.py -rd data/split_bird_dataset/train -nd data/npz_bird_dataset/train -cp data/bird_dataset_classmap.txt
# convert val data to npz
$ python data_preparation/_2_convert_dataset_to_npz.py -rd data/split_bird_dataset/val -nd data/npz_bird_dataset/val -cp data/bird_dataset_classmap.txt
# convert test data to npz
$ python data_preparation/_2_convert_dataset_to_npz.py -rd data/split_bird_dataset/test -nd data/npz_bird_dataset/test -cp data/bird_dataset_classmap.txt
```

### Convert Data to tfrecords

```shell
# convert train npz files into train tfrecord, select NUM_SHARDS so that each shard has around 200 imgs
$ python data_preparation/_3_convert_npz_tfrecord.py -nd data/npz_bird_dataset/train -td data/tfrecord_bird_dataset/train -ns NUM_SHARDS
# convert train npz files into val tfrecord, select NUM_SHARDS so that each shard has around 200 imgs
$ python data_preparation/_3_convert_npz_tfrecord.py -nd data/npz_bird_dataset/val -td data/tfrecord_bird_dataset/val -ns NUM_SHARDS
# convert train npz files into test tfrecord, select NUM_SHARDS so that each shard has around 200 imgs
$ python data_preparation/_3_convert_npz_tfrecord.py -nd data/npz_bird_dataset/test -td data/tfrecord_bird_dataset/test -ns NUM_SHARDS
```

## Model Selection

## Setting data paths and model hyper-parameters

## Model Training

## Model Testing

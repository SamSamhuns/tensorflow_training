import os

# os environments must be set at the beginning of the file top use GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OMP_NUM_THREADS"] = "15"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 1, 2, 3, 4, 5, 6, 7
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # prevent tf from using full GPU memory

import glob
import logging
import tensorflow as tf
from datetime import datetime
from easydict import EasyDict as edict
from models_info import model_info_dict
import tensorflow_model_optimization as tfmot
from tensorflow.python.client import device_lib


logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

tf.executing_eagerly()
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False)
# tf.config.experimental.enable_mlir_graph_optimization()  # gives a channel depth error

local_device_protos = device_lib.list_local_devices()
print("Available devices: ", [x.name for x in local_device_protos])

# ##################### Set Paths ######################
train_tfr_data_dir = "data/sample_tfrecord_bird_dataset/train"
val_tfr_data_dir = "data/sample_tfrecord_bird_dataset/val"
test_image_data_dir = "data/split_bird_dataset/test"


training_dataset = glob.glob(os.path.join(train_tfr_data_dir, "*.tfrecord"))
validation_dataset = glob.glob(os.path.join(val_tfr_data_dir, "*.tfrecord"))

dt = datetime.now().strftime("%Y%m%d-%H%M%S")
scalar_logs = "logs/" + dt + "/scalars"
image_logs = "logs/" + dt + "/images"
prune_logs = "logs/" + dt + "/prune"
image_log_writer = tf.summary.create_file_writer(image_logs)
# ##############################################################################

# ########## choose model ###########
# current_model = edict(model_info_dict['densenet121'])
# current_model = edict(model_info_dict['EfficientNetB6'])
# current_model = edict(model_info_dict['InceptionResNetV2'])
# current_model = edict(model_info_dict['InceptionV3'])
# current_model = edict(model_info_dict['Xception'])
# current_model = edict(model_info_dict['facenet'])
# current_model = edict(model_info_dict['MobileNetV3Large'])
current_model = edict(model_info_dict['MobileNetV2'])

# ############################### Set Parameters ###############################
tfrecord_parallel_read = 20
total_classes = 265
batch_size = 32
val_batch_size = 32
warm_start = False  # False means cold start, starting training from scratch
warm_start_point = 'retraining_model'  # irrelevant if warm_start is False
epochs = 1
val_freq = 1  # Also checkpointing epoch
total_training_samples = 37345
workers = 100  # model.fit workers
# for adam lr=0.001; SGD with momentum, lr=0.01/2 with momentum = 0.9
learning_rate = 0.01  # use a small learning rate if model is clustered say 1e-5
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
# settings for testing
model_test_ckpt = "checkpoints/inference_model"
class_map_txt_path = "data/bird_dataset_classmap.txt"
# ##############################################################################

# ############################# model optimization #############################
clustering = False
cluster_layers = False
is_model_already_clustered = False
num_clusters = 4
# options: LINEAR, DENSITY_BASED, RANDOM
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization.DENSITY_BASED
cluster_layer_names = ['conv2d_1']

during_training_quantization = False
quantize_layers = False
post_training_quantization = False
is_model_already_quantized = False
quantize_layer_names = ['conv2d_1']

pruning = False
prune_layers = False
is_model_already_pruned = False
prune_layer_names = ['conv2d_1']
# ##############################################################################

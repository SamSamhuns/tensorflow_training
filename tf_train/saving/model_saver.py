from functools import partial

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tf_train.pipelines import representative_data_gen as repr_data_gen


def get_flops(tf_model_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        with session.as_default():
            _ = tf.keras.models.load_model(tf_model_path)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops


def print_flops_n_train_tm(config, model, model_savepath, train_time):
    day = train_time // (24 * 3600)
    train_time = train_time % (24 * 3600)
    hour = train_time // 3600
    train_time %= 3600
    mins = train_time // 60
    train_time %= 60
    secs = train_time

    config.logger.info(f"Printing stats for model at {model_savepath}")
    config.logger.info(f"\tTotal Flops : {get_flops(model_savepath)}")
    config.logger.info(
        f"\tTraining Time: {day}:{hour}:{mins}:{secs} (d:h:m:s)")
    config.logger.info(f"\tTotal Parameters: {model.count_params()}")


def save_model(model, train_time, config):
    # savepath for retraining and inference model
    retrain_path = infer_path = config.save_dir

    # tf.lite.OpsSet.SELECT_TF_OPS
    # Add the above line to to make lite models support tf ops. Binary size will increase.
    during_train_qnt = config["optimization"]["quantize"]["during_training_quantization"]
    post_train_qtn = config["optimization"]["quantize"]["post_training_quantization"]
    qtn_layers = config["optimization"]["quantize"]["quantize_layers"]
    use_clustering = config["optimization"]["cluster"]["use_clustering"]
    cluster_layers = config["optimization"]["cluster"]["cluster_layers"]
    prune_layers = config["optimization"]["prune"]["prune_layers"]

    if (during_train_qnt or qtn_layers):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        model1 = converter.convert()
        with open(config.save_dir / "qa.tflite", "wb") as f:
            f.write(model1)
        retrain_path /= "qa_retrain_model"
        infer_path /= "qa_infer_model"
        tf.keras.models.save_model(model, retrain_path, include_optimizer=True)
        tf.keras.models.save_model(model, infer_path, include_optimizer=False)
        print_flops_n_train_tm(config, model, infer_path, train_time)
    elif post_train_qtn:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = partial(
            repr_data_gen, config=config)
        # Option 1
        # converter.target_spec.supported_types = [tf.float16]
        # Option 2
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS]
        model1 = converter.convert()
        with open(config.save_dir / "ptq.tflite", "wb") as f:
            f.write(model1)
    elif use_clustering or cluster_layers:
        retrain_path /= "cluster_retrain_model"
        infer_path /= "cluster_infer_model"
        tf.keras.models.save_model(model, retrain_path, include_optimizer=True)
        model = tfmot.clustering.keras.strip_clustering(model)
        tf.keras.models.save_model(model, infer_path, include_optimizer=False)
        print_flops_n_train_tm(config, model, infer_path, train_time)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        model1 = converter.convert()
        with open(config.save_dir / "cluster.tflite", "wb") as f:
            f.write(model1)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = partial(
            repr_data_gen, config=config)
        # Option 1
        # converter.target_spec.supported_types = [tf.float16]
        # Option 2
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS]
        model1 = converter.convert()
        with open(config.save_dir / "cluster_ptq.tflite", "wb") as f:
            f.write(model1)
    elif prune_layers:
        retrain_path /= "prune_retrain_model"
        infer_path /= "prune_infer_model"
        tf.keras.models.save_model(model, retrain_path, include_optimizer=True)
        model = tfmot.sparsity.keras.strip_pruning(model)
        tf.keras.models.save_model(model, infer_path, include_optimizer=False)
        print_flops_n_train_tm(config, model, infer_path, train_time)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        model1 = converter.convert()
        with open(config.save_dir / "prune.tflite", "wb") as f:
            f.write(model1)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = partial(
            repr_data_gen, config=config)
        # Option 1
        # converter.target_spec.supported_types = [tf.float16]
        # Option 2
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS]
        model1 = converter.convert()
        with open(config.save_dir / "prune_ptq.tflite", "wb") as f:
            f.write(model1)
    else:
        infer_path /= "infer_model"
        retrain_path /= "retrain_model"
        tf.keras.models.save_model(model, retrain_path, include_optimizer=True)
        tf.keras.models.save_model(model, infer_path, include_optimizer=False)
        print_flops_n_train_tm(config, model, infer_path, train_time)

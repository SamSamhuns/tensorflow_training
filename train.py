# set env ars from .env before importing any python libraries
from dotenv import load_dotenv
load_dotenv(".env")

import io
import math
import argparse
import collections
from datetime import datetime
from functools import partial
from contextlib import redirect_stdout

import tensorflow as tf
import tensorflow_model_optimization as tfmot

import tf_train.loss as module_loss
import tf_train.logging as module_log
import tf_train.metric as module_metric
import tf_train.optimizer as module_optim
from tf_train.model import get_model
from tf_train.utils import write_json
from tf_train.saving import save_model
from tf_train.config_parser import ConfigParser
from tf_train.model_optimization import optimize_model
from tf_train.pipelines import train_input_fn, val_input_fn


def train(config):
    config.setup_logger('train')

    # mixed precision training https://www.tensorflow.org/guide/mixed_precision, default=float32
    tf.keras.mixed_precision.set_global_policy(
        config["mixed_precision_global_policy"])
    config.logger.info(f"mixed_precision global_policy set to: {tf.keras.mixed_precision.global_policy()}")

    config.logger.info(f"TF executing eagerly: {tf.executing_eagerly()}")
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(False)
    # tf.config.experimental.enable_mlir_graph_optimization()  # gives a channel depth error

    ngpu_avai = len(tf.config.list_physical_devices('GPU'))
    config.logger.info(
        f"Available devices: {[x.name for x in tf.config.list_physical_devices()]}")
    config.logger.info(f"Num GPUs used: {ngpu_avai}")

    if ngpu_avai > 0:
        config.logger.info("Training on GPU")
        mirrored_strategy = tf.distribute.MirroredStrategy(
            devices=[f"/gpu:{dev}" for dev in range(ngpu_avai)])
    else:
        config.logger.info("Training on CPU")
        mirrored_strategy = tf.distribute.MirroredStrategy(
            devices=["/cpu:0"])
    # Convert and infer from pb file https://medium.com/@pipidog/how-to-convert-your-keras-models-to-tensorflow-e471400b886a
    with mirrored_strategy.scope():

        optimizer = config.init_ftn("optimizer", module_optim)()
        loss_weights = config["loss_weights"]
        loss = [config.init_ftn(["loss", _loss], module_loss, from_logits=config.model.gives_logits)()
                for _loss in config["loss"]]
        metrics = [config.init_ftn(["train_metrics", _metric], module_metric)()
                   for _metric in config["train_metrics"]]

        callbacks = []
        for cb_obj_name in config["callbacks"]:
            if cb_obj_name == "tensorboard_callback" and config["trainer"]["use_tensorboard"]:
                callback = config.init_obj(
                    ["callbacks", cb_obj_name], module_log,
                    log_dir=config["trainer"]["tf_scalar_logs"],
                    train_input_fn=partial(train_input_fn, config=config),
                    image_log_writer=tf.summary.create_file_writer(config["trainer"]["tf_image_logs"]))
            elif cb_obj_name == "ckpt_callback":
                callback = config.init_obj(
                    ["callbacks", cb_obj_name], tf.keras.callbacks,
                    filepath=config.save_dir / config["trainer"]["ckpt_fmt"])
            elif cb_obj_name == "epoch_log_lambda_callback":
                log_file = open(config.log_dir / "info.log",
                                mode='a', buffering=1)
                callback = config.init_obj(
                    ["callbacks", cb_obj_name], tf.keras.callbacks,
                    on_epoch_end=lambda epoch, logs: log_file.write(
                        f"epoch: {epoch}, loss: {logs['loss']}, accuracy: {logs['accuracy']}, "
                        f"val_loss: {logs['val_loss']}, val_accuracy: {logs['val_accuracy']}\n"),
                    on_train_end=lambda logs: log_file.close())
            elif cb_obj_name == "update_initial_epoch_callback":
                def _update_initial_epoch(epoch):
                    config["trainer"]["initial_epoch"] = epoch
                    write_json(config._config, config.save_dir / 'config.json')
                callback = config.init_obj(
                    ["callbacks", cb_obj_name], tf.keras.callbacks,
                    on_epoch_end=lambda epoch, logs: _update_initial_epoch(epoch))
            else:
                callback = config.init_obj(
                    ["callbacks", cb_obj_name], tf.keras.callbacks)
            callbacks.append(callback)

        # precedence is given to cli -r/--resume over json config resume_checkpoint
        resume_ckpt = None
        if config.resume is not None:
            resume_ckpt = config.resume
        elif config.resume is None and config["resume_checkpoint"] is not None:
            resume_ckpt = config["resume_checkpoint"]

        if resume_ckpt is None:
            config.logger.info("Cold Starting")
            model = get_model(config)
            tf.config.optimizer.set_jit("autoclustering")
            model.compile(optimizer=optimizer, loss=loss,
                          loss_weights=loss_weights, metrics=metrics)
        else:
            config.logger.info(f"Warm starting from {resume_ckpt}")
            with tfmot.quantization.keras.quantize_scope(), \
                    tfmot.clustering.keras.cluster_scope(), \
                    tfmot.sparsity.keras.prune_scope():
                model = tf.keras.models.load_model(resume_ckpt, compile=True)
            # To change anything except learning rate, recompilation is required
            model.optimizer.lr = config["optimizer"]["args"]["learning_rate"]
            model, callbacks = optimize_model(
                model, loss, optimizer, loss_weights, metrics, callbacks, config)

        # stream model summary to logger
        f = io.StringIO()
        with redirect_stdout(f):
            model.summary()
        model_summary = f.getvalue()
        config.logger.info(model_summary)

        train_steps = math.ceil(
            config["data"]["num_train_samples"] / config["data"]["train_bsize"])
        val_steps = math.ceil(
            config["data"]["num_val_samples"] / config["data"]["val_bsize"])

        start_time = datetime.today().timestamp()
        model.fit(x=train_input_fn(config),
                  epochs=config["trainer"]["epochs"],
                  steps_per_epoch=train_steps,
                  validation_data=val_input_fn(config),
                  validation_steps=val_steps,
                  validation_freq=config["trainer"]["val_freq"],
                  callbacks=callbacks, verbose=config["trainer"]["verbosity"],
                  initial_epoch=config["trainer"]["initial_epoch"],
                  workers=config["trainer"]["num_workers"],
                  use_multiprocessing=config["trainer"]["use_multiproc"])
        training_time = datetime.today().timestamp() - start_time
        save_model(model, training_time, config)

        # print model input, output shapes
        try:
            loaded_model = tf.keras.models.load_model(
                config.save_dir / "retrain_model")
            infer = loaded_model.signatures["serving_default"]
            config.logger.info(infer.structured_input_signature)
            config.logger.info(infer.structured_outputs)
        except Exception as e:
            config.logger.error(
                f"{e}. Could not get model input-output name and shapes.")


def main():
    parser = argparse.ArgumentParser(description='Tensorflow Training')
    # primary cli args
    parser.add_argument('--cfg', '--config', type=str, dest="config", default="config/train_image_clsf.json",
                        help='config file path (default: %(default)s)')
    parser.add_argument('--id', '--run_id', type=str, dest="run_id", default="train_" + datetime.now().strftime(r'%Y%m%d_%H%M%S'),
                        help='unique identifier for train process. Annotates train ckpts & logs. (default: %(default)s)')
    parser.add_argument('-r', '--resume', type=str, dest="resume", default=None,
                        help='path to resume ckpt. Overrides `resume_checkpoint` in config. (default: %(default)s)')

    # custom cli options to modify configuration from default values given in json file.
    # should be used to reset train params when resuming checkpoint, i.e. reducing LR
    OverrideArgs = collections.namedtuple(
        'OverrideArgs', 'flags dest help type target')
    options = [
        OverrideArgs(['--lr', '--learning_rate'],
                     dest="learning_rate", help="lr param to override that in config. (default: %(default)s)",
                     type=float, target='optimizer;args;learning_rate'),
        OverrideArgs(['--bs', '--train_bsize'],
                     dest="train_bsize", help="train bsize to override that in config. (default: %(default)s)",
                     type=int, target='data;train_bsize')
    ]
    config = ConfigParser.from_args(parser, options)
    train(config)


if __name__ == "__main__":
    main()

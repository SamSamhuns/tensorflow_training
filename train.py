# set env ars from .env before importing any python libraries
from dotenv import load_dotenv
load_dotenv(".env")

import io
import argparse
from datetime import datetime
from functools import partial
from contextlib import redirect_stdout

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.python.client import device_lib

import tf_train.loss as module_loss
import tf_train.logging as module_log
import tf_train.metric as module_metric
import tf_train.optimizer as module_optim
from tf_train.model import get_model
from tf_train.saving import save_model
from tf_train.config_parser import ConfigParser
from tf_train.preprocessing import train_input_fn, val_input_fn
from tf_train.model_optimization import optimize_model


def train(config):
    logger = config.get_logger('train')

    tf.executing_eagerly()
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(False)
    # tf.config.experimental.enable_mlir_graph_optimization()  # gives a channel depth error

    local_device_protos = device_lib.list_local_devices()
    ngpu_avai = len(tf.config.list_physical_devices('GPU'))
    logger.debug(f"Available devices: {[x.name for x in local_device_protos]}")
    logger.info(f"Num GPUs used: {ngpu_avai}")

    if ngpu_avai > 0:
        mirrored_strategy = tf.distribute.MirroredStrategy(
            devices=[f"/gpu:{dev}" for dev in range(ngpu_avai)])
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
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
                    filepath=config.save_dir / "{epoch:04d}")
            elif cb_obj_name == "epoch_log_lambda_callback":
                log_file = open(config.log_dir / "info.log",
                                mode='a', buffering=1)
                callback = config.init_obj(
                    ["callbacks", cb_obj_name], tf.keras.callbacks,
                    on_epoch_end=lambda epoch, logs: log_file.write(
                        f"epoch: {epoch}, loss: {logs['loss']}, accuracy: {logs['accuracy']}, "
                        f"val_loss: {logs['val_loss']}, val_accuracy: {logs['val_accuracy']}\n"),
                    on_train_end=lambda logs: log_file.close())
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
            logger.info("Cold Starting")
            model = get_model(config)
            tf.config.optimizer.set_jit(True)
            model.compile(optimizer=optimizer, loss=loss,
                          loss_weights=loss_weights, metrics=metrics)
        else:
            logger.info("Warm starting from " + resume_ckpt)
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
        logger.info(model_summary)

        start_time = datetime.today().timestamp()

        model.fit(train_input_fn(config),
                  initial_epoch=0, epochs=config["trainer"]["epochs"],
                  callbacks=callbacks, verbose=config["trainer"]["verbosity"],
                  validation_data=val_input_fn(config),
                  validation_freq=config["trainer"]["val_freq"],
                  workers=config["trainer"]["num_workers"], use_multiprocessing=True)
        training_time = datetime.today().timestamp() - start_time

        save_model(model, training_time, config)

        # print model input, output shapes
        try:
            loaded_model = tf.keras.models.load_model(
                config.save_dir / "retrain_model")
            infer = loaded_model.signatures["serving_default"]
            logger.info(infer.structured_input_signature)
            logger.info(infer.structured_outputs)
        except Exception as e:
            logger.error(
                f"{e}. Could not get model input-output name and shapes.")


def main():
    args = argparse.ArgumentParser(description='Tensorflow Training')
    args.add_argument('-cfg', '--config', default="config/train_image_clsf.json", type=str,
                      help='config file path (default: %(default)s)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to resume ckpt. Overrides `resume_checkpoint` in config. (default: %(default)s)')
    args.add_argument('-id', '--run_id', default="train_" + datetime.now().strftime(r'%Y%m%d_%H%M%S'), type=str,
                      help='unique identifier for train process. Annotates train ckpts & logs. (default: %(default)s)')
    config = ConfigParser.from_args(args)
    train(config)


if __name__ == "__main__":
    main()

from dotenv import load_dotenv

load_dotenv(".env")
# set env vars from .env before importing any python libraries
import tensorflow_model_optimization as tfmot
import tensorflow as tf
from omegaconf import OmegaConf
from contextlib import redirect_stdout
from functools import partial
from datetime import datetime
import argparse
import math
import io
import os
from tensorflow_training.pipelines import train_input_fn, val_input_fn
from tensorflow_training.model_optimization import optimize_model
from tensorflow_training.config_parser import ConfigParser
from tensorflow_training.saving import save_model
from tensorflow_training.model import get_model
import tensorflow_training.optimizer as module_optim
import tensorflow_training.metric as module_metric
import tensorflow_training.loss as module_loss
import tensorflow_training.logging as module_log


def train(config: ConfigParser):
    config.setup_logger("train")

    # mixed precision training https://www.tensorflow.org/guide/mixed_precision, default=float32
    tf.keras.mixed_precision.set_global_policy(config["mixed_precision_global_policy"])
    config.logger.info(
        f"mixed_precision global_policy set to: {tf.keras.mixed_precision.global_policy()}"
    )

    config.logger.info(f"TF executing eagerly: {tf.executing_eagerly()}")
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(False)
    # tf.config.experimental.enable_mlir_graph_optimization()  # gives a channel depth error

    ngpu_avai = len(tf.config.list_physical_devices("GPU"))
    config.logger.info(
        f"Available devices: {[x.name for x in tf.config.list_physical_devices()]}"
    )
    config.logger.info(f"Num GPUs used: {ngpu_avai}")

    if ngpu_avai > 0:
        config.logger.info("Training on GPU")
        mirrored_strategy = tf.distribute.MirroredStrategy(
            devices=[f"/gpu:{dev}" for dev in range(ngpu_avai)]
        )
    else:
        config.logger.info("Training on CPU")
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])

    with mirrored_strategy.scope():
        optimizer = config.init_ftn("optimizer", module_optim)()
        loss_weights = list(config["loss_weights"])
        loss = [
            config.init_ftn(
                ["loss", _loss], module_loss, from_logits=config.model.gives_logits
            )()
            for _loss in config["loss"]
        ]
        metrics = [
            config.init_ftn(["train_metrics", _metric], module_metric)()
            for _metric in config["train_metrics"]
        ]

        callbacks = []
        for cb_obj_name in config["callbacks"]:
            if (
                cb_obj_name == "tensorboard_callback"
                and config["trainer"]["use_tensorboard"]
            ):
                callback = config.init_obj(
                    ["callbacks", cb_obj_name],
                    module_log,
                    log_dir=config["trainer"]["tf_scalar_logs"],
                    train_input_fn=partial(train_input_fn, config=config),
                    image_log_writer=tf.summary.create_file_writer(
                        config["trainer"]["tf_image_logs"]
                    ),
                )
            elif cb_obj_name == "ckpt_callback":
                callback = config.init_obj(
                    ["callbacks", cb_obj_name],
                    tf.keras.callbacks,
                    filepath=os.path.join(
                        config.models_dir, config["trainer"]["ckpt_fmt"]
                    ),
                )
            elif cb_obj_name == "epoch_log_lambda_callback":
                log_file = open(
                    os.path.join(config.logs_dir, "info.log"),
                    mode="a",
                    buffering=1,
                    encoding="utf-8",
                )
                callback = config.init_obj(
                    ["callbacks", cb_obj_name],
                    tf.keras.callbacks,
                    on_epoch_end=lambda epoch, logs: log_file.write(
                        f"epoch: {epoch}, loss: {logs['loss']}, accuracy: {logs['accuracy']}, "
                        f"val_loss: {logs['val_loss']}, val_accuracy: {logs['val_accuracy']}\n"
                    ),
                    on_train_end=lambda logs: log_file.close(),
                )
            elif cb_obj_name == "update_initial_epoch_callback":

                def _update_initial_epoch(epoch):
                    config.trainer.initial_epoch = epoch
                    cfg = dict(config.config)
                    # save updated initial_epoch to config.yaml
                    save_root = os.path.join(
                        config.save_dir, config.experiment_name, config.run_id
                    )
                    OmegaConf.save(cfg, os.path.join(save_root, "config.yaml"))

                callback = config.init_obj(
                    ["callbacks", cb_obj_name],
                    tf.keras.callbacks,
                    on_epoch_end=lambda epoch, logs: _update_initial_epoch(epoch),
                )
            else:
                callback = config.init_obj(
                    ["callbacks", cb_obj_name], tf.keras.callbacks
                )
            callbacks.append(callback)

        # precedence is given to cli -r/--resume over json config resume_checkpoint
        resume_ckpt = None
        if config.resume_checkpoint is not None:
            resume_ckpt = config.resume_checkpoint

        if resume_ckpt is None:
            config.logger.info("Cold Starting")
            model = get_model(config)
            tf.config.optimizer.set_jit("autoclustering")
            model.compile(
                optimizer=optimizer,
                loss=loss,
                loss_weights=loss_weights,
                metrics=metrics,
            )
        else:
            config.logger.info(f"Warm starting from {resume_ckpt}")
            with (
                tfmot.quantization.keras.quantize_scope(),
                tfmot.clustering.keras.cluster_scope(),
                tfmot.sparsity.keras.prune_scope(),
            ):
                model = tf.keras.models.load_model(resume_ckpt, compile=True)
            # To change anything except learning rate, recompilation is required
            model.optimizer.lr = config["optimizer"]["args"]["learning_rate"]
            model, callbacks = optimize_model(
                model, loss, optimizer, loss_weights, metrics, callbacks, config
            )

        # stream model summary to logger
        f = io.StringIO()
        with redirect_stdout(f):
            model.summary()
        model_summary = f.getvalue()
        config.logger.info(model_summary)

        train_steps = math.ceil(
            config["data"]["num_train_samples"] / config["data"]["train_bsize"]
        )
        val_steps = math.ceil(
            config["data"]["num_val_samples"] / config["data"]["val_bsize"]
        )

        start_time = datetime.today().timestamp()
        model.fit(
            x=train_input_fn(config),
            epochs=config["trainer"]["epochs"],
            steps_per_epoch=train_steps,
            validation_data=val_input_fn(config),
            validation_steps=val_steps,
            validation_freq=config["trainer"]["val_freq"],
            callbacks=callbacks,
            verbose=config["trainer"]["verbosity"],
            initial_epoch=config["trainer"]["initial_epoch"],
            workers=config["trainer"]["num_workers"],
            use_multiprocessing=config["trainer"]["use_multiproc"],
        )
        training_time = datetime.today().timestamp() - start_time
    save_model(model, training_time, config)

    # print model input, output shapes
    try:
        loaded_model = tf.keras.models.load_model(
            os.path.join(config.models_dir, "retrain_model")
        )
        infer = loaded_model.signatures["serving_default"]
        config.logger.info(infer.structured_input_signature)
        config.logger.info(infer.structured_outputs)
    except Exception as e:
        config.logger.error(f"{e}. Could not get model input-output name and shapes.")


def main():
    parser = argparse.ArgumentParser(description="Tensorflow Training")
    # primary cli args
    parser.add_argument(
        "--cfg",
        "--config",
        type=str,
        dest="config",
        default="config/train_image_clsf.yaml",
        help="YAML config file path (default: %(default)s)",
    )
    parser.add_argument(
        "--id",
        "--run_id",
        type=str,
        dest="run_id",
        default="train_" + datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        help="Unique identifier for train process. Annotates train checkpoints & logs. (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--override",
        type=str,
        nargs="+",
        dest="override",
        default=None,
        help="Override config params. Must match keys in YAML config. "
        "e.g. -o seed=1 dataset.type=NewDataType model.layers=[64,128,256] model.layers[2]=512 (default: %(default)s)",
    )
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        type=str,
        dest="resume_checkpoint",
        help="Path to resume checkpoint. Overrides `resume_checkpoint` in config.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Run training in verbose mode (default: %(default)s)",
    )

    # additional args
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        dest="learning_rate",
        default=None,
        help="lr param to override that in config. (default: %(default)s)",
    )
    parser.add_argument(
        "--bs",
        "--train_bsize",
        type=int,
        dest="train_bsize",
        default=None,
        help="train bsize to override that in config. (default: %(default)s)",
    )
    args = parser.parse_args()

    # To override key-value params from YAML file,
    # match the YAML kv structure for any additional args above
    # keys-val pairs can have nested structure separated by colons
    yaml_modification = {
        "trainer.args.resume_checkpoint": args.resume_checkpoint,
        "optimizer.args.learning_rate": args.learning_rate,
        "data.train_bsize": args.train_bsize,
    }
    # get custom omegaconf DictConfig-like obj
    cfg = ConfigParser.from_args(args, yaml_modification)
    train(cfg)


if __name__ == "__main__":
    main()

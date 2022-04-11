# It's generally better to finetune with pruning, clustering,
# and quantizing as opposed to training from scratch.
# Try pruning the later layers instead of the first layers.
# Avoid pruning critical layers (e.g. attention mechanism).
from functools import partial
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tf_train.utils import rgetattr


# @tf.function
def apply_quantization_to_layer(layer, quantize_layer_names):
    if (layer.name in quantize_layer_names):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

    # Quantize a type of layer
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer


# @tf.function
def apply_clustering_to_layer(layer, cluster_layer_names, clustering_params):
    if (layer.name in cluster_layer_names):
        return tfmot.clustering.keras.cluster_weights(layer, **clustering_params)
    return layer


# @tf.function
def apply_pruning_to_layer(layer, prune_layer_names):
    if (layer.name in prune_layer_names):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer


# @tf.function
def optimize_model(model, loss, optimizer, loss_weights, metrics, callbacks, config):
    # ###################### quantization configurations #######################
    during_train_qnt = config["optimization"]["quantize"]["during_training_quantization"]
    is_model_alr_qnt = config["optimization"]["quantize"]["is_model_already_quantized"]
    qtn_layer = config["optimization"]["quantize"]["quantize_layers"]

    if during_train_qnt and (not is_model_alr_qnt):
        model = tfmot.quantization.keras.quantize_model(model)
        loss = {'quant_' + key: val for key, val in loss.items()}
        model.compile(optimizer=optimizer, loss=loss,
                      loss_weights=loss_weights, metrics=metrics)
    if qtn_layer and (not is_model_alr_qnt):
        model = tf.keras.models.clone_model(
            model, clone_function=partial(
                apply_quantization_to_layer,
                quantize_layer_names=config["optimization"]["prune"]["quantize_layer_names"]), )
        model = tfmot.quantization.keras.quantize_apply(model)
        # Uncomment only if last layer is being clustered
        # loss = {'quant_' + key: val for key, val in loss.items()}
        model.compile(optimizer=optimizer, loss=loss,
                      loss_weights=loss_weights, metrics=metrics)

    # ####################### clustering configurations ########################
    clustering_params = {
        'number_of_clusters': config["optimization"]["cluster"]["num_clusters"],
        'cluster_centroids_init': rgetattr(
            tfmot,
            config["optimization"]["cluster"]["CentroidInitialization"])
    }
    use_clustering = config["optimization"]["cluster"]["use_clustering"]
    is_model_alr_clustered = config["optimization"]["cluster"]["is_model_already_clustered"]
    cluster_layers = config["optimization"]["cluster"]["cluster_layers"]

    if use_clustering and (not is_model_alr_clustered):
        model = tfmot.clustering.keras.cluster_weights(
            model, **clustering_params)
        loss = {'cluster_' + key: val for key, val in loss.items()}
        model.compile(optimizer=optimizer, loss=loss,
                      loss_weights=loss_weights, metrics=metrics)
    if cluster_layers and (not is_model_alr_clustered):
        model = tf.keras.models.clone_model(
            model, clone_function=partial(
                apply_clustering_to_layer,
                cluster_layer_names=config["optimization"]["cluster"]["cluster_layer_names"]), )
        # Uncomment only if last layer is being clustered
        # loss = {'cluster_' + key: val for key, val in loss.items()}

        model.compile(optimizer=optimizer, loss=loss,
                      loss_weights=loss_weights, metrics=metrics)

    # ########################## pruning configurations ########################
    total = config["data"]["num_train_samples"]
    bsize = config["data"]["train_bsize"]
    end_step = np.ceil(total / bsize).astype(np.int32) * \
        config["trainer"]["epochs"]
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.80,
                                                                 begin_step=0,
                                                                 end_step=end_step)}
    use_pruning = config["optimization"]["prune"]["use_pruning"]
    is_model_alr_pruned = config["optimization"]["prune"]["is_model_already_pruned"]
    prune_layers = config["optimization"]["prune"]["prune_layers"]

    if use_pruning and (not is_model_alr_pruned):
        model = tfmot.sparsity.keras.prune_low_magnitude(
            model, **pruning_params)
        loss = {'prune_low_magnitude_' + key: val for key, val in loss.items()}
        model.compile(optimizer=optimizer, loss=loss,
                      loss_weights=loss_weights, metrics=metrics)
        pruning_callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
                             tfmot.sparsity.keras.PruningSummaries(log_dir=config["trainer"]["tf_scalar_logs"]), ]
        callbacks.append(pruning_callbacks)
    if prune_layers:
        model = tf.keras.models.clone_model(
            model, clone_function=partial(
                apply_pruning_to_layer,
                prune_layer_names=config["optimization"]["prune"]["prune_layer_names"]))
        # Uncomment only if last layer is being clustered
        # loss = {'prune_' + key: val for key, val in loss.items()}

        model.compile(optimizer=optimizer, loss=loss,
                      loss_weights=loss_weights, metrics=metrics)
        pruning_callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
                             tfmot.sparsity.keras.PruningSummaries(log_dir=config["trainer"]["tf_prune_logs"]), ]
        callbacks.append(pruning_callbacks)
    return model, callbacks

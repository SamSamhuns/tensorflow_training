import logging
import tensorflow as tf


def get_model(config):
    logging.info(f"Chosen Model: {config.model.type}")
    logging.info(f"Model Input Shape (H, W, C) {config.model.args.input_shape}")
    inputs = tf.keras.Input(
        shape=config.model.args.input_shape, name='input_img')
    # if include_top is set to false, the default classification heads are omited and the model is
    # only a feature extractor, so the input_tensor=Input(shape=(W, H, C)) has to be specified
    main_model = config.model.module(**config.model.args,
                                     classes=config["data"]["num_classes"])

    main_model.trainable = config["trainable_feat_backbone"]
    try:
        if main_model.trainable is False and config.model.args.include_top is False:
            logging.info(
                "NOTE: model is missing top so cannot be used for classification.")
    except Exception:
        pass  # include_top might not be present

    # check if an additional final layer/model is used, i.e. for more dense prediction heads
    final_module = config.model.final_module
    if final_module is not None:
        final = final_module(num_classes=config["data"]["num_classes"])
        isLayer = isinstance(final, tf.keras.layers.Layer)
        isModel = isinstance(final, tf.keras.Model)
        if isLayer is False and isModel is False:
            msg = f"{final} must be of type tf.keras.Layer or  tf.keras.Model but found type {type(final)} instead"
            logging.error(msg)
            raise TypeError(msg)
        outputs = main_model(inputs)
        outputs = final(outputs)
    else:
        outputs = main_model(inputs)
    return tf.keras.Model(inputs=[inputs], outputs=[outputs], name=config.model.type)

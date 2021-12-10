import tensorflow as tf


def get_facenet_model(classes=100,
                      checkpoint_path="custom_models/facenet_weights/facenet_keras.h5",
                      train_final_clsf_layer_only=False,
                      num_final_dense_layers=1,
                      **kwargs):
    """
    Returns a tf.keras.models.Model with an input_shape of (160, 160, 3)
    """
    warm_model = tf.keras.models.load_model(checkpoint_path, compile=True)

    # freeze feat extraction layers and only train classification heads
    warm_model.trainable = not train_final_clsf_layer_only

    # add classification heads
    if num_final_dense_layers == 2:
        out1 = tf.keras.layers.Dense(
            256,
            activation='relu',
            name='final_dense_1')(warm_model.layers[-1].output)
        out2 = tf.keras.layers.Dense(
            classes,
            activation='softmax',
            name='final_dense_2')(out1)
    elif num_final_dense_layers == 1:
        out2 = tf.keras.layers.Dense(
            classes,
            activation='softmax',
            name='final_dense_1')(warm_model.layers[-1].output)

    return tf.keras.models.Model(
        inputs=warm_model.inputs,
        outputs=out2)


def preprocess_input(image_rgb):
    img_whitened = tf.image.per_image_standardization(image_rgb)
    return img_whitened

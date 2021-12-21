import tensorflow as tf


def SGD(learning_rate=0.01, **kwargs):
    return tf.keras.optimizers.SGD(learning_rate=learning_rate, **kwargs)


def Adam(learning_rate=0.001, **kwargs):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate, **kwargs)

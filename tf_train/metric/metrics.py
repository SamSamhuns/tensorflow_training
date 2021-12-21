import tensorflow as tf


def CategoricalAccuracy(**kwargs):
    return tf.keras.metrics.CategoricalAccuracy(**kwargs)


def TopKCategoricalAccuracy(**kwargs):
    return tf.keras.metrics.TopKCategoricalAccuracy(**kwargs)

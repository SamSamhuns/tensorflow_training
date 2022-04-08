import tensorflow as tf


class Classifier(tf.keras.Model):
    """
    Fully connected layer for classification with a separate acgivation layer
    with a dtype of float32 to allow for mixed precision training
    """

    def __init__(self, num_classes: int):
        super(Classifier, self).__init__()

        blocks = []
        # last classification and activation layer
        # sep to set last layer dtype to f32 when using mixed_precision
        blocks.append(tf.keras.layers.Dense(num_classes, dtype='float32'))
        blocks.append(tf.keras.layers.Activation('softmax', dtype='float32'))
        self.blocks = tf.keras.Sequential([*blocks])

    def call(self, x):
        return self.blocks(x)

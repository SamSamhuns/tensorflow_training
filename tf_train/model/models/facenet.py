import tensorflow as tf


def load_facenet_model(input_shape, classes, weights):
    """
    Returns a tf.keras.models.Model with an input_shape of (160, 160, 3)
    """
    model = tf.keras.models.load_model(weights, compile=True)
    return model


class FacenetPred(tf.keras.Model):
    """
    Classic fully connected neural network that downsamples features by half every layer
    """

    def __init__(self, num_classes: int, init_units=512, num_final_blocks: int = 2):
        super(FacenetPred, self).__init__()

        blocks = []
        units = init_units
        # add num_final_blocks-1 blocks before final block
        for i in range(num_final_blocks - 1):
            dense = tf.keras.layers.Dense(
                units,
                activation='relu')
            blocks.append(dense)
            units //= 2

        # last classification layer
        blocks.append(tf.keras.layers.Dense(num_classes, activation='softmax'))
        self.blocks = tf.keras.Sequential([*blocks])

    def call(self, x):
        return self.blocks(x)


def preprocess_input(image_rgb):
    img_whitened = tf.image.per_image_standardization(image_rgb)
    return img_whitened

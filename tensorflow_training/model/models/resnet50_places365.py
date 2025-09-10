import tensorflow as tf

MEAN, STD = tf.constant([0.485, 0.456, 0.406]), tf.constant([0.229, 0.224, 0.225])


def load_resnet50_places365_model(input_shape, classes, weights):
    """
    Returns a tf.keras.models.Model with an input_shape of(244, 224, 3)
    """
    model = tf.keras.models.load_model(weights, compile=True)
    return model


class Resnet50Places365Pred(tf.keras.Model):
    """
    Classic fully connected neural network that downsamples features by half every layer
    """

    def __init__(self, num_classes: int, init_units=512, num_final_blocks: int = 2):
        super(Resnet50Places365Pred, self).__init__()

        blocks = []
        units = init_units
        # add num_final_blocks-1 blocks before final block
        for _ in range(num_final_blocks - 1):
            dense = tf.keras.layers.Dense(
                units,
                activation='relu')
            blocks.append(dense)
            units //= 2

        # last classification and activation layer
        # sep to set last layer dtype to f32 when using mixed_precision
        blocks.append(tf.keras.layers.Dense(num_classes, dtype='float32'))
        blocks.append(tf.keras.layers.Activation('softmax', dtype='float32'))
        self.blocks = tf.keras.Sequential([*blocks])

    def call(self, x):
        return self.blocks(x)


def preprocess_input(image_rgb):
    image_rgb = image_rgb / 255.
    img_whitened = (image_rgb - MEAN) / STD
    return img_whitened

import tensorflow as tf


class Residual(tf.keras.Model):
    """
    This module looks like what you find in the original resnet or IC paper
    (https://arxiv.org/pdf/1905.05928.pdf), except that it's based on MLP, not CNN.
    If you flag `only_MLP` as True, then it won't use any batch norm, dropout, or
    residual connections
    """

    def __init__(self, num_features: int, dropout: float,
                 add_residual: bool, add_IC: bool, i: int, j: int):
        super(Residual, self).__init__()

        self.num_features = num_features
        self.add_residual = add_residual
        self.add_IC = add_IC
        self.i = i
        self.j = j

        if (not ((self.i == 0) and (self.j == 0))) and self.add_IC:
            self.norm_layer1 = tf.keras.layers.BatchNormalization()
            self.dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.linear1 = tf.keras.layers.Dense(
            num_features, activation='relu', name="linear1")

        if self.add_IC:
            self.norm_layer2 = tf.keras.layers.BatchNormalization()
            self.dropout2 = tf.keras.layers.Dropout(rate=dropout)
        self.linear2 = tf.keras.layers.Dense(
            num_features, activation=None, name="linear2")
        self.relu2 = tf.keras.layers.ReLU()

    def call(self, x):

        identity = out = x

        if (not ((self.i == 0) and (self.j == 0))) and self.add_IC:
            out = self.norm_layer1(out)
            out = self.dropout1(out)
        out = self.linear1(out)

        if self.add_IC:
            out = self.norm_layer2(out)
            out = self.dropout2(out)
        out = self.linear2(out)

        if self.add_residual:
            out += identity

        out = self.relu2(out)
        return out


class DownSample(tf.keras.Model):
    """
    This module is an MLP, where the number of output features is lower than
    that of input features. If you flag `only_MLP` as False, it'll add norm
    and dropout
    """

    def __init__(self, in_features: int, out_features: int, dropout: float,
                 add_IC: bool):
        super(DownSample, self).__init__()
        assert in_features > out_features

        self.in_features = in_features
        self.out_features = out_features
        self.add_IC = add_IC

        if self.add_IC:
            self.norm_layer = tf.keras.layers.BatchNormalization()
            self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.linear = tf.keras.layers.Dense(
            out_features, activation='relu', name="linear")

    def call(self, x):
        out = x

        if self.add_IC:
            out = self.norm_layer(out)
            out = self.dropout(out)
        out = self.linear(out)
        return out


class ResMLP(tf.keras.Model):
    """
    MLP with optinally batch norm, dropout, and residual connections. I got
    inspiration from the original ResNet paper and https://arxiv.org/pdf/1905.05928.pdf.

    Downsampling is done after every block so that the features can be encoded
    and compressed.
    """

    def __init__(self, dropout: float, num_residuals_per_block: int, num_blocks: int, num_classes: int,
                 num_initial_features: int, reduce_in_features: int, add_residual: bool = True, add_IC: bool = True):
        super(ResMLP, self).__init__()

        blocks = []
        # input feature space reduction layer, acts as encoder layer
        # if reduce_feat_num is not None, reduce input features with downsampling instead of residual block
        if reduce_in_features is not None:
            blocks.append(DownSample(
                num_initial_features, reduce_in_features, dropout, add_IC))
        else:
            reduce_in_features = num_initial_features

        for i in range(num_blocks):
            blocks.extend(self._create_block(
                reduce_in_features, dropout, num_residuals_per_block, add_residual, add_IC, i))
            reduce_in_features //= 2

        # last classification layer
        blocks.append(tf.keras.layers.Dense(num_classes))
        self.blocks = tf.keras.Sequential([*blocks])

    def _create_block(self, in_features: int, dropout: float,
                      num_residuals_per_block: int, add_residual: bool,
                      add_IC: bool, i: int) -> list:
        block = []
        for j in range(num_residuals_per_block):
            block.append(Residual(in_features, dropout,
                                  add_residual, add_IC, i, j))
        block.append(DownSample(
            in_features, in_features // 2, dropout, add_IC))
        return block

    def call(self, x):
        return self.blocks(x)


class FullyConnectedNet(tf.keras.Model):
    """
    Classic fully connected neural network that downsamples features by half every layer
    """

    def __init__(self, num_blocks: int, num_classes: int,
                 num_initial_features: int, reduce_in_features: int, **kwargs):
        super(FullyConnectedNet, self).__init__()

        blocks = []
        # input feature space reduction layer, acts as encoder layer
        # if reduce_feat_num is not None, reduce input features with downsampling instead of residual block
        if reduce_in_features is not None:
            blocks.append(tf.keras.layers.Dense(reduce_in_features, activation='relu'))
        else:
            reduce_in_features = num_initial_features

        for i in range(num_blocks):
            blocks.extend(self._create_block(reduce_in_features))
            reduce_in_features //= 2

        # last classification layer
        blocks.append(tf.keras.layers.Dense(num_classes))
        self.blocks = tf.keras.Sequential([*blocks])

    def _create_block(self, in_features: int) -> list:
        block = []
        block.append(tf.keras.layers.Dense(in_features // 2, activation='relu'))
        return block

    def call(self, x):
        return self.blocks(x)


if __name__ == "__main__":
    res_model = Residual(512, 0.1, True, True, 0, 0)
    res_model.build(input_shape=(None, 512))
    res_model.summary()
    print(res_model(tf.ones((1, 512))).shape)

    ds_model = DownSample(512, 256, 0.1, True)
    ds_model.build(input_shape=(None, 512))
    ds_model.summary()
    print(ds_model(tf.ones((1, 512))).shape)

    res_mlp_model = ResMLP(0.1, 3, 2, 4, 512, 256, True, True)
    res_mlp_model.build(input_shape=(None, 512))
    res_mlp_model.summary()
    print(res_mlp_model(tf.ones((1, 512))).shape)

    fcnn_model = FullyConnectedNet(dropout=0.05, num_residuals_per_block=1, num_blocks=4,
                                   num_classes=4, num_initial_features=512, reduce_in_features=256,
                                   add_residual=True, add_IC=True)
    fcnn_model.build(input_shape=(None, 512))
    fcnn_model.summary()
    print(fcnn_model(tf.ones((1, 512))).shape)

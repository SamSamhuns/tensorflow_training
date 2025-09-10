import tensorflow as tf


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    """
    def on_batch_end(self, batch, logs=None):
        ...  # log on batch end =
        super().on_batch_end(batch, logs)
    """
    def __init__(self, log_dir, train_input_fn, image_log_writer, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.train_input_fn = train_input_fn
        self.image_log_writer = image_log_writer

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        logs.update({'total_reg_loss': tf.reduce_sum(self.model.losses)})

        # To print model input
        iter_obj = iter(self.train_input_fn(bsize=3))
        features, _ = next(iter_obj)
        image = features['input_img']

        with self.image_log_writer.as_default():
            tf.summary.image('Train/output', image, step=epoch,
                             max_outputs=3, description="descrip")
        super().on_epoch_end(epoch, logs)

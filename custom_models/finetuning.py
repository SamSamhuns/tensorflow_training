import tensorflow as tf


def custom_model():
    print("TRAINING MODE: \n",
          "Chosen Model: ", current_model.module, '\n',
          "Model Input Width, Height, Channel: ",
          current_model.in_w,
          current_model.in_h,
          current_model.in_c)
    # Variable-length sequence of ints
    input_img = tf.keras.Input(
        shape=(current_model.in_w, current_model.in_h, current_model.in_c), name='input_img')
    # if include_top is set to false, the default classification heads are omited and the model is
    # only a feature extractor, so the input_tensor=Input(shape=(W, H, C)) has to be specified
    this_model = current_model.module(include_top=False,
                                      input_shape=(
                                          current_model.in_w, current_model.in_h, current_model.in_c),
                                      weights="imagenet",  # None, imagenet
                                      pooling="avg",  # None, avg, max
                                      classes=total_classes)
    this_model.trainable = False
    x1 = this_model(input_img)
    out = tf.keras.layers.Dense(
        total_classes,
        activation='softmax',
        name='final_dense')(x1)
    # x1 = tf.compat.v1.Print(x1, [tf.math.reduce_sum(x1)])
    return tf.keras.Model(inputs=[input_img], outputs=[out])

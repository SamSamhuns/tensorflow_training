import os
import argparse
import tensorflow as tf
from _1_std_headers import current_model


class CustomModule(tf.Module):
    def __init__(self, model, **kwargs):
        super(CustomModule, self).__init__(**kwargs)
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, current_model.in_w, current_model.in_h, 3), dtype=tf.uint8)])
    def update_signature(self, input_images):  # inp_images is the input name
        # x = tf.image.central_crop(inp_images, 0.8)
        x = tf.cast(input_images, tf.float32)
        x = current_model.parent_module.preprocess_input(x)
        output = self.model(x)
        return {"predictions": output}


def convert_h5_to_savedmodel(h5_model_path, savedmodel_export_dir):
    os.makedirs(savedmodel_export_dir, exist_ok=True)

    model = tf.keras.models.load_model(h5_model_path)
    module = CustomModule(model)
    tf.saved_model.save(module, savedmodel_export_dir, signatures={
                        "serving_default": module.update_signature})
    print("Export complete.")


def main():
    parser = argparse.ArgumentParser("""
        Convert img dataset into npz files with embedded class label info:
        eg. python _2_convert_dataset_to_npz -id data/split_bird_dataset/train
                                             -td data/tfrecord_bird_dataset/train
                                             -ns 100""")
    parser.add_argument('-h5', '--h5_model_path',
                        type=str, default="checkpoints/inference_model.h5",
                        help="h5 model to convert. (default %(default)s)")
    parser.add_argument('-e', '--savedmodel_export_dir',
                        type=str, default="checkpoints/1/model.savedmodel",
                        help="Dir path where the savedmodel will be exported to. (default %(default)s)")
    args = parser.parse_args()
    convert_h5_to_savedmodel(args.h5_model_path, args.savedmodel_export_dir)


if __name__ == "__main__":
    main()

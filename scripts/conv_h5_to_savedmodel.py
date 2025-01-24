import os
import argparse
import traceback

import tensorflow as tf
from easydict import EasyDict as edict

from tf_train.utils.common import read_json
from tf_train.model.models_info import model_info_dict


def create_custom_module(model, model_info):
    h, w, c = model_info.args.input_shape

    class CustomModule(tf.Module):
        def __init__(self, model, **kwargs):
            super(CustomModule, self).__init__(**kwargs)
            self.model = model

        @tf.function(input_signature=[tf.TensorSpec(shape=(None, w, h, c), dtype=tf.uint8)])
        def update_signature(self, input_images):  # inp_images is the input name
            # x = tf.image.central_crop(inp_images, 0.8)
            x = tf.cast(input_images, tf.float32)
            x = model_info.parent_module.preprocess_input(x)
            output = self.model(x)
            return {"predictions": output}

    return CustomModule(model)


def convert_h5_to_savedmodel(config_path: str, h5_model_path: str, savedmodel_export_dir: str):
    os.makedirs(savedmodel_export_dir, exist_ok=True)

    config = read_json(config_path)
    model_info = edict(model_info_dict[config["arch"]])

    model = tf.keras.models.load_model(h5_model_path)
    custom_module = create_custom_module(model, model_info)

    try:
        tf.saved_model.save(custom_module, savedmodel_export_dir,
                            signatures={"serving_default": custom_module.update_signature})
        print("\x1b[6;30;42m" + f"Export to {savedmodel_export_dir} successful." + "\x1b[0m")
    except Exception as e:
        traceback.print_exc()
        print(e)
        print("\x1b[6;30;41m" + "Export failed." + "\x1b[0m")


def main():
    parser = argparse.ArgumentParser("""Export h5 model into savedmodel format""")
    parser.add_argument('-cfg', '--config', required=True, type=str,
                        help='config file path (default: %(default)s)')
    parser.add_argument('-h5', '--h5_model_path',
                        type=str, required=True,
                        help="source h5 model path for conversion")
    parser.add_argument('-e', '--savedmodel_export_dir',
                        type=str, default="checkpoints/1/model.savedmodel",
                        help="savedmodel export dir path. (default %(default)s)")
    args = parser.parse_args()
    convert_h5_to_savedmodel(args.config,
                             args.h5_model_path,
                             args.savedmodel_export_dir)


if __name__ == "__main__":
    main()

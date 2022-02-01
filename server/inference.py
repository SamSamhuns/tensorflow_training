import os
import cv2
import time
import numpy as np
from functools import partial

from triton_server.utils import extract_data_from_media, get_client_and_model_metadata_config
from triton_server.utils import parse_model_grpc, get_inference_responses
from triton_server.utils import FlagConfig, resize_maintaining_aspect


FLAGS = FlagConfig()


def preprocess(img, width=640, height=480, new_type=np.uint8):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_maintaining_aspect(img, width, height).astype(new_type)
    return img


def postprocess(results, output_names):
    predictions = []
    pred_idx_2_classes = ["c1", "c2", "c3"]  # add class names in correct order

    for output_name in output_names:
        logits = results.as_numpy(output_name)
        softmax_confs = np.exp(logits) / np.sum(np.exp(logits))
        class_idx = np.argmax(logits)
        if softmax_confs[:, class_idx] >= FLAGS.threshold:
            predictions.append(pred_idx_2_classes[class_idx])

    if not predictions:
        predictions = ["No faces detected."]
    return predictions


def run_inference(
        input_file,
        model_name,
        threshold,
        inference_mode="image",
        port=8081,
        debug=True):
    FLAGS.media_filename = input_file
    FLAGS.model_name = model_name
    FLAGS.threshold = threshold
    FLAGS.inference_mode = inference_mode
    FLAGS.debug = debug
    FLAGS.result_save_dir = None  # set to None prevent saving
    FLAGS.model_version = ""  # empty str means use latest
    FLAGS.protocol = "grpc"
    FLAGS.url = f"127.0.0.1:{port}"
    FLAGS.verbose = False
    FLAGS.classes = 0  # classes must be set to 0
    FLAGS.batch_size = 1
    FLAGS.fixed_input_width = None
    FLAGS.fixed_input_height = None
    start_time = time.time()

    if FLAGS.result_save_dir is not None:
        FLAGS.result_save_dir = os.path.join(
            FLAGS.save_result_dir, f"{FLAGS.model_name}")
        os.makedirs(FLAGS.result_save_dir, exist_ok=True)
    if FLAGS.debug:
        print(f"Running model {FLAGS.model_name}")

    model_info = get_client_and_model_metadata_config(FLAGS)
    if model_info == -1:  # error getting model info
        raise Exception("Model could not be loaded in the server")
    triton_client, model_metadata, model_config = model_info

    # input_name, output_name, format, dtype are all lists
    max_batch_size, input_name, output_name, h, w, c, format, dtype = parse_model_grpc(
        model_metadata, model_config.config)

    # check for dynamic input shapes
    if h == -1:
        h = FLAGS.fixed_input_height
    if w == -1:
        w = FLAGS.fixed_input_width

    filenames = []
    if isinstance(FLAGS.media_filename, str) and os.path.isdir(FLAGS.media_filename):
        filenames = [
            os.path.join(FLAGS.media_filename, f)
            for f in os.listdir(FLAGS.media_filename)
            if os.path.isfile(os.path.join(FLAGS.media_filename, f))
        ]
    else:
        filenames = [
            FLAGS.media_filename,
        ]
    filenames.sort()

    nptype_dict = {"UINT8": np.uint8, "FP32": np.float32, "FP16": np.float16}
    # Important, make sure the first input is the input image
    image_input_idx = 0
    preprocess_dtype = partial(
        preprocess, width=w, height=h, new_type=nptype_dict[dtype[image_input_idx]])
    # all_reqested_images_orig will be [] if FLAGS.result_save_dir is None
    image_data, all_reqested_images_orig, all_req_imgs_orig_size = extract_data_from_media(
        FLAGS, preprocess_dtype, filenames)
    if len(image_data) == 0:
        raise Exception("Image data is missing. Aborting inference")

    trt_inf_data = (triton_client, input_name,
                    output_name, dtype, max_batch_size)
    # if a model has multiple inputs, pass inputs as a list
    image_data_list = [image_data]
    # get inference results
    responses = get_inference_responses(
        image_data_list, FLAGS, trt_inf_data)

    counter = 0
    final_result_list = []
    for response in responses:
        pred_classes = postprocess(response, output_name)
        final_result_list.append(pred_classes)

        # display boxes on image array
        if FLAGS.result_save_dir is not None:
            drawn_img = all_reqested_images_orig[counter]
            drawn_img = resize_maintaining_aspect(drawn_img, w, h)
        counter += 1
    if FLAGS.inference_mode == "video":
        final_result_list = [pred[0] for pred in final_result_list]
    if FLAGS.debug:
        print(f"Time to process {counter} image(s)={time.time()-start_time}")

    return final_result_list

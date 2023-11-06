import os
import pathlib
import cv2
import PIL

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
import tensorflow_hub as hub

from IPython.display import display, Javascript
from base64 import b64decode, b64encode
import io
import html
import time

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

from moviepy.editor import VideoFileClip

PATH_TO_LABELS = "faster_rcnn_model/License-Plate-Violations-xPDX_label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True
)


def run_inference_for_single_image(model, image, live_cam):
    # convert image into numpy
    image = np.asarray(image)
    # print('Converted image into numpy type:', type(image))

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # print('Converted numpy into tensor format:', input_tensor)

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    if not live_cam:
        start_time = time.time()
        output_dict = model(input_tensor)
        end_time = time.time()
        print(f"Inference time: {np.ceil(end_time-start_time)} seconds per frame")

    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop("num_detections"))  # 300

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.

    output_dict = {
        key: value[0, :num_detections].numpy() for key, value in output_dict.items()
    }

    output_dict["num_detections"] = num_detections

    # detection_classes should be ints.
    output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int64)

    return output_dict


def run_inference_video(model, video_path, live_cam):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res = (int(width), int(height))

        # save detected video
        # Initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MPEG")  # codec
        out = cv2.VideoWriter("output_videos/detected_output.avi", fourcc, 30.0, res)
        frame = None

        while True:
            try:
                is_success, image_np = cap.read()
            except cv2.error:
                continue

            if not is_success:
                break

            # Actual detection.
            output_dict = run_inference_for_single_image(model, image_np, live_cam)

            # Visualization of the results of a detection.
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict["detection_boxes"],
                output_dict["detection_classes"],
                output_dict["detection_scores"],
                category_index,
                instance_masks=output_dict.get("detection_masks_reframed", None),
                use_normalized_coordinates=True,
                line_thickness=8,
            )

            out.write(image_np)

        out.release()

    cap.release()

def run(file_path, file_name, close_loading, start, stop, update_text):
    global PATH_TO_LABELS, category_index  
    PATH_TO_LABELS = "faster_rcnn_model/License-Plate-Violations-xPDX_label_map.pbtxt"
    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=True
    )

    model_handle = "faster_rcnn_model/saved_model"

    update_text("DETECTION STATUS: Loading Model...")
    hub_model = hub.load(model_handle)
    print("model loaded!")
    # Inference on captured video
    video_path = file_path
    start()
    update_text("DETECTION STATUS: Detection Currently Running...")
    
    run_inference_video(hub_model, video_path, live_cam=False)
    stop()
    """ Download "detected_output.avi" video and play in your laptop video player """

    # Input video path
    save_path = "output_videos/detected_output.avi"

    # Compressed video path
    compressed_path = f"output_videos/faster_rcnn_{file_name}"

    video = VideoFileClip(save_path)
    video.write_videofile(compressed_path, codec="libx264")
    close_loading(compressed_path)

    # os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

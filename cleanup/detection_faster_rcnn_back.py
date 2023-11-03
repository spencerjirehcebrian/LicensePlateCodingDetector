from base64 import b64encode
from IPython.display import HTML
import cv2
import warnings
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import time
import tensorflow as tf
import pathlib
import os
import os.path


def run(file_path, file_name, close_loading):
    print('working')
    #os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)

    #tf.get_logger().setLevel("ERROR")  # Suppress TensorFlow logging (2)

    # Enable GPU dynamic memory allocation
    # gpus = tf.config.experimental.list_physical_devices("GPU")
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    #     print(gpu)

    # Preparing the test images

    # # IMAGE_PATHS = [
    # #     "export-test-1-final/test/test (1).jpg", "export-test-1-final/test/test (2).jpg", "export-test-1-final/test/test (3).jpg", "export-test-1-final/test/test (4).jpg", "export-test-1-final/test/test (5).jpg", "export-test-1-final/test/test (6).jpg", "export-test-1-final/test/test (7).jpg"]
    # IMAGE_PATHS = []
    # # change range if you only want to test a few images
    # for i in range(1, 31):
    #     IMAGE_PATHS.append("faster_rccn_model/detect/detect (" + str(i) + ").jpg")

    PATH_TO_MODEL_DIR = "faster_rccn_model"

    # # checking contents of the image_paths
    # for x in range(len(IMAGE_PATHS)):
    #     print(IMAGE_PATHS[x])

    # Preparing the labels

    LABEL_FILENAME = "License-Plate-Violations-xPDX_label_map.pbtxt"
    PATH_TO_LABELS = "/faster_rccn_model/License-Plate-Violations-xPDX_label_map.pbtxt"

    # Load the model
    # ~~~~~~~~~~~~~~
    # Next we load the downloaded model

    PATH_TO_SAVED_MODEL = "faster_rccn_model/saved_model/saved_model.pb"

    print("Loading model...", end="")
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done! Took {} seconds".format(elapsed_time))

    # Load label map data (for plotting)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Label maps correspond index numbers to category names, so that when our convolution network
    # predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
    # functions, but anything that returns a dictionary mapping integers to appropriate string labels
    # would be fine.

    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=True
    )

    # Putting everything together
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The code shown below loads an image, runs it through the detection model and visualizes the
    # detection results, including the keypoints.
    #
    # Note that this will take a long time (several minutes) the first time you run this code due to
    # tf.function's trace-compilation --- on subsequent runs (e.g. on new images), things will be
    # faster.

    def visualise_on_image(image, bboxes, labels, scores, thresh):
        (h, w, d) = image.shape
        for bbox, label, score in zip(bboxes, labels, scores):
            if score > thresh:
                xmin, ymin = int(bbox[1] * w), int(bbox[0] * h)
                xmax, ymax = int(bbox[3] * w), int(bbox[2] * h)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"{label}: {int(score*100)} %",
                    (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
        return image

    if __name__ == "__main__":  
        # Load the model
        print("Loading saved model ...")
        detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        print("Model Loaded!")

        # Video Capture (video_file)
        video_capture = cv2.VideoCapture(file_path)
        start_time = time.time()

        frame_width = int(video_capture.get(3))
        frame_height = int(video_capture.get(4))
        # fps = int(video_capture.get(5))
        size = (frame_width, frame_height)

        # Initialize video writer
        result = cv2.VideoWriter(
            f"{file_name}", cv2.VideoWriter_fourcc(*"MJPG"), 15, size
        )

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Unable to read video / Video ended")
                break

            frame = cv2.flip(frame, 1)
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            # The model expects a batch of images, so also add an axis with `tf.newaxis`.
            input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

            # Pass frame through detector
            detections = detect_fn(input_tensor)

            # Set detection parameters

            score_thresh = 0.4  # Minimum threshold for object detection
            max_detections = 1

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            scores = detections["detection_scores"][0, :max_detections].numpy()
            bboxes = detections["detection_boxes"][0, :max_detections].numpy()
            labels = (
                detections["detection_classes"][0, :max_detections]
                .numpy()
                .astype(np.int64)
            )
            labels = [category_index[n]["name"] for n in labels]

            # Display detections
            visualise_on_image(frame, bboxes, labels, scores, score_thresh)

            end_time = time.time()
            fps = int(1 / (end_time - start_time))
            start_time = end_time
            cv2.putText(
                frame,
                f"FPS: {fps}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                1,
            )
            # cv2_imshow(frame)

            # Write output video
            result.write(frame)

        video_capture.release()

import argparse
import sys
from functools import lru_cache

import select

import cv2
import numpy as np
import psutil
import subprocess
import csv
import time

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

last_detections = []
frame_number = 0

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)

    set_infer_time(metadata=metadata)

    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def set_infer_time(metadata: dict):
    kpi_info = IMX500.get_kpi_info(metadata=metadata)

    if kpi_info is not None:
        dnn_runtime_ms, dsp_runtime_ms = kpi_info

        row[1] = round(dnn_runtime_ms + dsp_runtime_ms, 5)


def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    detections = last_results
    if detections is None:
        return
    labels = get_labels()

    with MappedArray(request, stream) as m:
        for detection in detections:
            # Only get people:
            if labels[int(detection.category)] != "person":
                continue

            x, y, w, h = detection.box

            # =====================ADDED=====================
            row[4] += labels[int(detection.category)] + ", "
            row[5] += f'{detection.conf:.2f}' + ", "
            row[6] += f"{tuple(map(int, detection.box))}, "
            #################################################

            # Draw detection box
            cv2.rectangle(m.array, (x+10, y+10), (x + w, y + h), (0, 255, 0, 0), thickness=2)

        row[7] = row[7].rstrip(", ")

        # =====================ADDED=====================
        cv2.putText(m.array, str(frame_number), (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        frame = m.array
        if frame.size > 0:
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            out.write(frame)
        #################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()


if __name__ == "__main__":
    cpu_usage = psutil.cpu_percent()
    temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
    temp_c = float(temp_output.replace("temp=", "").replace("'C\n", ""))

    args = get_args()

    # =====================ADDED=====================
    name = "11N_IMX_30"
    vid = "Vid13"

    csv_log_filename = f"logs/{name}/{vid}/orig.csv"
    csv_log_file = open(csv_log_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_log_file, delimiter=';')

    csv_writer.writerow(["Frame",
                         "Inference YOLO",
                         "CPU%",
                         "Temp(C)",
                         "Detected Objects",
                         "Confidence",
                         "Boxes",
                         "Timestamp"])

    #################################################

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        # print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate})

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections

    # =====================ADDED=====================
    text = frame_number
    font = cv2.FONT_HERSHEY_COMPLEX
    text_size = cv2.getTextSize(str(frame_number), font, 1,2)[0]

    # Setup VideoWriter for saving the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    test_frame = picam2.capture_array()
    height, width = test_frame.shape[:2]
    fps = intrinsics.inference_rate
    out = cv2.VideoWriter(f'logs/{name}/{vid}/orig.mp4', fourcc, fps, (width, height))
    #################################################

    while True:
        start_time = time.time()

        row = [""] * 7
        row[0] = f'{frame_number}'
        row[2] = f'{cpu_usage}'
        row[3] = f'{temp_c}'

        cpu_usage = psutil.cpu_percent()
        temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        temp_c = float(temp_output.replace("temp=", "").replace("'C\n", ""))

        last_results = parse_detections(picam2.capture_metadata())

        row[7] = f'{round(time.time() - start_time, 2)}'

        csv_writer.writerow(row)
        frame_number += 1

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == 'q':
                print("Quitting")
                break

    cv2.destroyAllWindows()
    out.release()
    csv_log_file.close()

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

import csv
import psutil
import subprocess
import time

model_name = "yolov8n_ncnn_model"

name = "V8N_NCNN"  # Make a folder with this name
vid = "Vid13"  # Make a folder with this name

csv_log_filename = f"logs/{name}/{vid}/orig.csv"
csv_log_file = open(csv_log_filename, mode='w', newline='')
csv_writer = csv.writer(csv_log_file, delimiter=';')

csv_writer.writerow(["Frame",
                     "Inference YOLO",
                     "CPU%",
                     "Temp(C)",
                     "Detected Objects",
                     "Confidence",
                     "boxes",
                     "TimeStamp"])

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 1280)  # Change as you prefer
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

model = YOLO(model_name)
model.conf = 0.3

# Setup VideoWriter for saving the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
test_frame = picam2.capture_array()
width, height = test_frame.shape[:2]
fps = 5
out = cv2.VideoWriter(f'logs/{name}/{vid}/orig.mp4', fourcc, fps, (width, height))

frame_number = 0

font = cv2.FONT_HERSHEY_COMPLEX
text_size = cv2.getTextSize(str(frame_number), font, 1,2)[0]

try:
    while True:
        start_time = time.time()

        cpu_usage = psutil.cpu_percent()
        temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        temp_c = float(temp_output.replace("temp=", "").replace("'C\n", ""))

        frame = picam2.capture_array()

        results = model(frame, verbose=False, retina_masks=False, classes=[0])

        # ====Plot Detection (Preview Video)====
        annotated_frame = results[0].plot(labels=False)

        text_x = annotated_frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10 - 10
        cv2.putText(annotated_frame, str(frame_number), (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Camera", annotated_frame)
        #######################

        boxes = results[0].boxes
        class_names = "None"

        # Get the detected boxes and names
        if boxes.cls.shape[0] > 0:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            class_names = [model.names[c] for c in class_ids]

        cpu_usage = psutil.cpu_percent()
        temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        temp_c = float(temp_output.replace("temp=", "").replace("'C\n", ""))

        csv_writer = csv.writer(csv_log_file, delimiter=';')

        if class_names != "None":
            class_names = ", ".join(class_names)

        # Write frame to video file
        out.write(annotated_frame)

        total_inference_time = (
                results[0].speed["preprocess"] + results[0].speed["inference"] + results[0].speed["postprocess"])

        # Save results
        csv_writer.writerow([frame_number, 
                             round(total_inference_time, 2),
                             f'{cpu_usage}',
                             f'{temp_c}',
                             class_names,
                             boxes.conf.tolist(),
                             boxes.xyxy.tolist(),
                             round(time.time() - start_time, 2)
                            ])

        if cv2.waitKey(1) == ord("q"):
            break

        frame_number += 1


finally:
    cv2.destroyAllWindows()
    out.release()
    picam2.stop()
    csv_log_file.close()
# Summary
This GitHub repository is part of a thesis project comparing off-board (CPU) models with on-board (IMX500) models.
The comparison is done with YOLO11n and YOLOv8n on MOT17 videos.

## RPI
The RPI directory contains all files needed to convert the models and start using the RPI camera to detect people.
Please convert and export the models first before recording.
In order to use the IMX version use the following:
```
python3 on-board.py --model path/to/network.rpk --fps {NUMBER} --bbox-normalization --ignore-dash-labels --bbox-order xy --labels path/to/labels.txt
```

## Mac
This directory contains a Python file to do offline detection (no camera needed).

## Evaluation
This directory contains the following:
- getPixelLoc.py -> Used to get the pixel location of the monitor.
- transform{IMX}\{NCNN}DetectedBBoxes.py -> Used to translate the bounding boxes so they align with the MOT17 video. Use the video to get the starting frame and the end frame.
- MOT17_image_to_vid.py -> Used to convert the MOT17 images, when downloading the dataset, into a video.
- detections_per_frame.py -> The annotation format is different for the MOT17 ground truth, this converts it so that each frame has all detections next to it.
- finalEval.py -> Evaluate final results of RPI.
- resultsMacOffline.py -> Evaluate final results of Mac.

# License and Copyright
This project incorporates code from the [Raspberry Pi project](https://github.com/raspberrypi/picamera2), which is licensed under the BSD 2-Clause License.
The file in 'examples/imx500/imx500_object_detection_demo.py' has been used and is now named 'on-board.py', with modifications tailored to the project.

from ultralytics import YOLO
import pandas as pd

# Load model
model = YOLO('yolo11n.pt')

# Define input video and output CSV
video_path = 'path/to/MOT17/vid.mp4'
# Create output file
output_csv = 'path/to/new/file.csv'

video_fps = 25  # Change this for each video: 30.88 (Vid9), 30 (Vid11), or 25 (Vid13)

# Make video 5 fps - remove this to get the highest detection speed
interval = round(video_fps / 5)

# Run tracking
results = model.track(source=video_path, stream=True, classes=[0])  # class 0 = person

# Prepare list for rows
csv_rows = []
frame_id = -1

for result in results:
    frame_id += 1

    # Make 5 FPS
    if frame_id % interval != 0:
        continue

    if result.boxes is None:
        continue

    for box in result.boxes:
        if int(box.cls[0]) != 0:
            continue
        bbox = box.xyxy[0].tolist()
        conf = float(box.conf[0]) if box.conf is not None else -1

        csv_rows.append([
            frame_id + 1,
            *[round(v, 2) for v in bbox],
            round(conf, 4)
        ])

# Save to CSV
df = pd.DataFrame(csv_rows, columns=['frame', 'x1', 'y1', 'x2', 'y2', 'confidence'])
df.to_csv(output_csv, index=False)
print(f'Detections saved to {output_csv}')
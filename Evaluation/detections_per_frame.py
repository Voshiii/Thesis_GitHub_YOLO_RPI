import os

mot_path = "path/to/MOT17-13-SDP"  # Change to your name
gt_file = os.path.join(mot_path, "gt", "gt.txt")

# Create file to save annotations
log_file = open(f'new/path/to/MOT17-13-SDP_detections.txt', 'w')

annotations = {}

with open(gt_file, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        frame_id = int(parts[0])
        x, y, w, h = map(float, parts[2:6])
        class_id = int(parts[6])

        if class_id != 1:
            continue  # Only take 'person' class
        if frame_id not in annotations:
            annotations[frame_id] = []
        annotations[frame_id].append((int(x), int(y), int(w), int(h)))

for item in annotations:
    log_file.write(f'{item} {annotations[item]}\n')

log_file.close()


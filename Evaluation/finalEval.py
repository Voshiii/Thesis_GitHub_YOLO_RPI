import csv
import ast
import cv2
import numpy as np

model = "11N_IMX_30"
vid = "Vid13"

# VID 09 => 30.88 fps
# VID 11 => 30 fps
# VID 13 => 25 fps

def load_ground_truth(file_path, fps=25):
    gt_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split(maxsplit=1)
            frame_id = int(parts[0])
            boxes = ast.literal_eval(parts[1])

            # I need to convert to correct format x, y, w, h -> x1, y1, x2, y2
            gt_boxes = [(*box[:2], box[0] + box[2], box[1] + box[3]) for box in boxes]
            timestamp = frame_id / fps
            gt_data[timestamp] = gt_boxes

    return gt_data


def load_detections(csv_path):
    det_data = {}

    # This is used to get the accumulated time for detection
    det_timestamp = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # Skip header

        for row in reader:
            try:
                timestamp = float(row[9])
                cell = row[7].strip()
                if cell and cell != '[]':
                    boxes = ast.literal_eval(cell)
                    if not det_timestamp:
                        det_data[timestamp] = [tuple(box) for box in boxes]
                        det_timestamp.append(timestamp)
                    else:
                        det_data[timestamp + det_timestamp[-1]] = [tuple(box) for box in boxes]
                        det_timestamp.append(timestamp + det_timestamp[-1])
                else:
                    if not det_timestamp:
                        det_data[timestamp] = []
                        det_timestamp.append(timestamp)
                    else:
                        det_data[timestamp + det_timestamp[-1]] = []
                        det_timestamp.append(timestamp + det_timestamp[-1])

            except (ValueError, IndexError):
                print("Error occurred loading detections")
                continue

    return det_data


def find_closest_gt_timestamp(gt_data, detection_timestamps):
    timestamps = sorted(gt_data.keys())
    # Compares all timestamps and takes the closes
    closest_ts = min(timestamps, key=lambda ts: abs(ts - detection_timestamps))
    return closest_ts


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def evaluate(gt_data, det_data, iou_threshold=0.5, offset=0, visualize=False, img_size=(1920, 1080)):
    TP, FP, FN = 0, 0, 0
    frame_idx = 0

    for det_ts, det_boxes in det_data.items():
        frame_idx += 1
        gt_ts = find_closest_gt_timestamp(gt_data, det_ts + offset)
        gt_boxes = gt_data.get(gt_ts, [])

        matched = [False] * len(gt_boxes)

        # Visualization
        if visualize:
            canvas = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

            # Green: Ground truth
            for box in gt_boxes:
                cv2.rectangle(canvas, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            # Red: Detected Boxes
            for box in det_boxes:
                cv2.rectangle(canvas, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(canvas, f"Frame {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Create png with both detections
            cv2.imwrite(f"new/path/to/saved/frame_{frame_idx:03d}.png", canvas)

        # Matching
        for det in det_boxes:
            match_found = False
            for i, gt in enumerate(gt_boxes):
                if not matched[i] and iou(det, gt) >= iou_threshold:
                    matched[i] = True
                    TP += 1
                    match_found = True
                    break
            if not match_found:
                FP += 1
        FN += matched.count(False)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {'TP': TP, 'FP': FP, 'FN': FN, 'Precision': precision, 'Recall': recall, 'F1': f1}


gt_boxes = load_ground_truth(f'path/to/MOT17/gt.txt')
det_boxes = load_detections(f'path/to/detection.csv')

log_file = open(f'path/to/saved/results.txt', 'w')

results = evaluate(gt_boxes, det_boxes, visualize=True)

log_file.write(f'{results}\n')

print(results)

log_file.close()

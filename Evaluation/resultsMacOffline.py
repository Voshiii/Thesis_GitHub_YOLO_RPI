import pandas as pd


def iou(boxA, boxB):
    # Compute Intersection over Union between two boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def evaluate(gt_path, pred_csv, video_fps, iou_thresh=0.5):
    # Load ground-truth
    gt = pd.read_csv(
        gt_path, header=None,
        names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    )
    gt = gt[gt['class'] == 1]  # class 1 is person in MOTChallenge
    gt['x2'] = gt['x'] + gt['w']
    gt['y2'] = gt['y'] + gt['h']
    gt['box'] = gt[['x','y','x2','y2']].values.tolist()

    # Load predictions
    det = pd.read_csv(pred_csv)
    det['box'] = det[['x1','y1','x2','y2']].values.tolist()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Sort the frames
    all_frames = sorted(set(gt['frame']).union(set(det['frame'])))

    # Remove if video should not be converted to 5 FPS
    interval = round(video_fps / 5)

    for frame_id in all_frames:
        # Remove if video should not be converted to 5 FPS
        if frame_id % interval != 1:
            continue

        gt_boxes = gt[gt['frame'] == frame_id]['box'].tolist()
        pred_boxes = det[det['frame'] == frame_id]['box'].tolist()

        matched_gt = set()
        matched_pred = set()

        for pi, pbox in enumerate(pred_boxes):
            for gi, gbox in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                if iou(pbox, gbox) >= iou_thresh:
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    break

        tp = len(matched_pred)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Change to save in txt file or csv
    print(f"TP: {total_tp}")
    print(f"FP: {total_fp}")
    print(f"FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


gt_path = f"path/to/MOT17/VID/gt.txt"
det_csv = "path/to/mac/results.csv"

# Change video FPS depending on the video
evaluate(gt_path, det_csv, video_fps=25)


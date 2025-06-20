import cv2
import csv
import numpy as np
import ast

# Use values from 'getPixelLoc.py'
x1, y1 = 83, 112  # top-left
x2, y2 = 1112, 85  # top-right
x3, y3 = 1112, 646  # bottom-right
x4, y4 = 140, 701  # bottom-left

points_src = np.array([
    [x1, y1],  # top-left
    [x2, y2],  # top-right
    [x3, y3],  # bottom-right
    [x4, y4],  # bottom-left
], dtype=np.float32)

# Monitor size
monitor_width = 1920
monitor_height = 1080

points_dst = np.array([
    [0, 0],
    [monitor_width - 1, 0],
    [monitor_width - 1, monitor_height - 1],
    [0, monitor_height - 1]
], dtype=np.float32)

# Get the transformation matrix
matrix = cv2.getPerspectiveTransform(points_src, points_dst)


# Transform a bounding box
def transform_bbox(bbox, matrix):
    x1, y1, x2, y2 = bbox
    corners = np.array([
        [x1, y1],
        [x2, y2]
    ], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, matrix)
    x1_new, y1_new = transformed[0][0]
    x2_new, y2_new = transformed[1][0]
    return [int(x1_new), int(y1_new), int(x2_new), int(y2_new)]


def transform_multiple_bboxes_cell(cell_str, matrix):
    # Parse string to list of lists
    bboxes = ast.literal_eval(cell_str)

    transformed = []
    for bbox in bboxes:
        new_bbox = transform_bbox(bbox, matrix)
        transformed.append(new_bbox)

    return str(transformed)


# ONLY USE NCNN
input_csv = f'path/to/orig.csv'
# Create output file
output_csv = f'new/path/to/converted.csv'

# CHANGE THESE FOR EACH VIDEO
video_playback_start = 16
video_playback_end = 81

with open(input_csv, 'r', newline='') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile, delimiter=';')
    writer = csv.writer(outfile, delimiter=';')

    header = next(reader)  # get header row
    writer.writerow(header)
    next(reader)

    for row in reader:
        try:
            # Only take during playback
            if int(row[0]) < video_playback_start:
                continue
            elif int(row[0]) > video_playback_end:
                break

            cell = row[6].strip()
            if cell == '[]' or not cell:
                writer.writerow(row)
                continue
            row[6] = transform_multiple_bboxes_cell(cell, matrix)
        except Exception as e:
            print(f"Skipping row due to error: {e}")
        writer.writerow(row)

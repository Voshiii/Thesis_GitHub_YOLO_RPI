import cv2
import os

# Define each video's parameters
videos = {
    "MOT17/train/MOT17-09-DPM": {"frames": 525, "duration": 17},
    "MOT17/train/MOT17-11-DPM": {"frames": 900, "duration": 30},
    "MOT17/train/MOT17-13-DPM": {"frames": 750, "duration": 30},
}


def calculate_fps(frames, duration):
    return frames / duration


def make_video(sequence_path, output_path, fps):
    img_dir = os.path.join(sequence_path, "img1")
    images = sorted([
        img for img in os.listdir(img_dir)
        if img.endswith(".jpg") or img.endswith(".png")
    ])

    if not images:
        raise ValueError(f"No image files found in {img_dir}")

    first_img = cv2.imread(os.path.join(img_dir, images[0]))
    height, width, _ = first_img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_name in images:
        frame = cv2.imread(os.path.join(img_dir, img_name))
        if frame is None:
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved: {output_path} ({fps:.2f} FPS)")


# Start
for vid_name, info in videos.items():
    fps = calculate_fps(info["frames"], info["duration"])
    input_folder = f'{vid_name}'
    output_file = f"{vid_name}.mp4"
    make_video(input_folder, output_file, fps)

import cv2

coordinates = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append((x, y))
        print(f"Clicked at: ({x}, {y})")
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Frame', param)


video_path = 'path/to/recorded/detection.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Add to go to next frame if the first frame is black
ret, frame = cap.read()

print("Coordinates order: top-left, top-right, bottom-right, bottom-left")

if ret:
    clone = frame.copy()
    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', click_event, clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("All clicked coordinates:", coordinates)

import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import matplotlib.pyplot as plt


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load COCO class names
with open("coco1.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv8 model
model = YOLO("../best_seg.pt")

# Open the video file (use video file or webcam, here using webcam)
cap = cv2.VideoCapture(0)
count = 0

# Initialize scatter data
scatter_x = []
scatter_y = []

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    # Ensure boxes exist in the results
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        # Check if tracking IDs exist before attempting to retrieve them
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = [-1] * len(boxes)  # Use -1 for objects without IDs

        masks = results[0].masks
        if masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = masks.xy
            overlay = frame.copy()

            for box, track_id, class_id, mask in zip(boxes, track_ids, class_ids, masks):
                # Convert mask points to integer
                c = class_names[class_id]

                x1, y1, x2, y2 = box

                # Calculate the centroid of the bounding box
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2

                # Append centroid to scatter data
                scatter_x.append(centroid_x)
                scatter_y.append(centroid_y)

                # Check if mask is not empty
                if mask.size > 0:
                    mask = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))  # Reshape mask to correct format

                    # Draw the bounding box and mask
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.fillPoly(overlay, [mask], color=(0, 0, 255))

                    # Draw the track ID and class label
                    cvzone.putTextRect(frame, f'{track_id}', (x2, y2), 1, 1)
                    cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

            alpha = 0.5  # Transparency factor (0 = invisible, 1 = fully visible)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Show the frame
    cv2.imshow("RGB", frame)

    # Check for keypress 'p' to plot scatter graph
    if cv2.waitKey(1) & 0xFF == ord("p"):
        plt.scatter(scatter_x, scatter_y, c='blue', alpha=0.5, label="Object Centroids")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.title("Scatter Plot of Detected Object Centroids")
        plt.legend()
        plt.show()

    if cv2.waitKey(1) & 0xFF == ord("esc"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

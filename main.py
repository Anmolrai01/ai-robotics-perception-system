from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

prev_time = 0

TARGET_CLASSES = ["person", "bottle", "cell phone"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # TRACK instead of predict
    results = model.track(frame, persist=True, conf=0.4)

    boxes = results[0].boxes
    names = results[0].names

    if boxes.id is not None:
        for box, track_id in zip(boxes, boxes.id):
            cls_id = int(box.cls[0])
            label = names[cls_id]

            if label in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(track_id)

                # Draw bounding box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                # Label with ID
                text = f"{label} ID:{track_id}"
                cv2.putText(frame, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("YOLO Tracking - Advanced", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
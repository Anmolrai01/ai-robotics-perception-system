from ultralytics import YOLO
import cv2
import time
import csv
from datetime import datetime
import pyttsx3

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO("yolov8n.pt")
model.fuse()

# -------------------------------
# TEXT TO SPEECH
# -------------------------------
engine = pyttsx3.init()
last_voice_time = 0

# -------------------------------
# VIDEO CAPTURE
# -------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# -------------------------------
# TARGET OBJECTS
# -------------------------------
TARGET_CLASSES = ["person", "bottle", "cell phone"]

# -------------------------------
# FPS VARIABLES
# -------------------------------
prev_time = 0

# -------------------------------
# CSV LOG FILE
# -------------------------------
csv_file = open("detection_log.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Time",
    "Object",
    "Track_ID",
    "Distance",
    "Region",
    "Threat_Level"
])

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Frame dimensions
    frame_height, frame_width, _ = frame.shape
    frame_center = frame_width // 2

    # Run tracking
    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        verbose=False
    )

    boxes = results[0].boxes
    names = results[0].names

    # -------------------------------
    # OBJECT COUNTERS
    # -------------------------------
    object_counts = {
        "person": 0,
        "bottle": 0,
        "cell phone": 0
    }

    # -------------------------------
    # DEFAULT STATES
    # -------------------------------
    warning_text = "PATH CLEAR"
    threat_level = "SAFE"

    # -------------------------------
    # DRAW REGIONS
    # -------------------------------
    cv2.line(frame, (frame_center - 150, 0),
             (frame_center - 150, frame_height),
             (255, 255, 0), 2)

    cv2.line(frame, (frame_center + 150, 0),
             (frame_center + 150, frame_height),
             (255, 255, 0), 2)

    cv2.putText(frame, "LEFT", (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,255,0), 2)

    cv2.putText(frame, "CENTER", (frame_center - 50, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,255,0), 2)

    cv2.putText(frame, "RIGHT", (frame_width - 120, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,255,0), 2)

    # -------------------------------
    # DETECTION LOOP
    # -------------------------------
    if boxes.id is not None:

        for box, track_id in zip(boxes, boxes.id):

            cls_id = int(box.cls[0])
            label = names[cls_id]

            if label not in TARGET_CLASSES:
                continue

            object_counts[label] += 1

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            track_id = int(track_id)

            # -------------------------------
            # DISTANCE ESTIMATION
            # -------------------------------
            width = x2 - x1

            if width > 250:
                distance = "NEAR"
            elif width > 120:
                distance = "MEDIUM"
            else:
                distance = "FAR"

            # -------------------------------
            # REGION DETECTION
            # -------------------------------
            center_x = (x1 + x2) // 2

            if center_x < frame_center - 150:
                region = "LEFT"

            elif center_x > frame_center + 150:
                region = "RIGHT"

            else:
                region = "CENTER"

            # -------------------------------
            # THREAT / DECISION LOGIC
            # -------------------------------
            if label == "person":

                if distance == "NEAR":

                    threat_level = "DANGER"

                    if region == "LEFT":
                        warning_text = "OBSTACLE LEFT -> TURN RIGHT"

                    elif region == "RIGHT":
                        warning_text = "OBSTACLE RIGHT -> TURN LEFT"

                    else:
                        warning_text = "STOP : OBJECT AHEAD"

                elif distance == "MEDIUM":

                    threat_level = "CAUTION"
                    warning_text = "SLOW DOWN"

                else:
                    threat_level = "SAFE"
                    warning_text = "PATH CLEAR"

            # -------------------------------
            # DRAW BOUNDING BOX
            # -------------------------------
            color = (0,255,0)

            if threat_level == "CAUTION":
                color = (0,255,255)

            if threat_level == "DANGER":
                color = (0,0,255)

            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          color,
                          2)

            # -------------------------------
            # LABEL TEXT
            # -------------------------------
            text = (
                f"{label} "
                f"ID:{track_id} "
                f"{distance} "
                f"{region}"
            )

            cv2.putText(frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

            # -------------------------------
            # CSV LOGGING
            # -------------------------------
            csv_writer.writerow([
                datetime.now().strftime("%H:%M:%S"),
                label,
                track_id,
                distance,
                region,
                threat_level
            ])

    # -------------------------------
    # MULTI-OBJECT LOGIC
    # -------------------------------
    total_persons = object_counts["person"]

    if total_persons >= 3:
        threat_level = "HIGH CROWD"
        warning_text = "MULTIPLE PEOPLE DETECTED"

    # -------------------------------
    # OBJECT COUNTS DISPLAY
    # -------------------------------
    count_text = (
        f"Persons: {object_counts['person']} | "
        f"Bottles: {object_counts['bottle']} | "
        f"Phones: {object_counts['cell phone']}"
    )

    cv2.putText(frame,
                count_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,0,0),
                2)

    # -------------------------------
    # THREAT LEVEL DISPLAY
    # -------------------------------
    cv2.putText(frame,
                f"STATUS: {threat_level}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2)

    # -------------------------------
    # DECISION DISPLAY
    # -------------------------------
    cv2.putText(frame,
                warning_text,
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2)

    # -------------------------------
    # FPS CALCULATION
    # -------------------------------
    curr_time = time.time()

    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0

    prev_time = curr_time

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2)

    # -------------------------------
    # VOICE ALERT
    # -------------------------------
    current_time = time.time()

    if threat_level == "DANGER":

        if current_time - last_voice_time > 5:

            engine.say("Warning obstacle ahead")
            engine.runAndWait()

            last_voice_time = current_time

    # -------------------------------
    # WINDOW
    # -------------------------------
    cv2.imshow(
        "AI Robotics Perception System",
        frame
    )

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
csv_file.close()
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
import csv
from datetime import datetime

# Paths
TRAINED_MODEL_DIR = "trained_models"
DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"

# Load Haarcascade Face Detector
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Load trained models and ID mappings
recognizers = {}
id_mappings = {}

for model_file in os.listdir(TRAINED_MODEL_DIR):
    if model_file.endswith("_model.yml"):
        section_name = model_file.replace("_model.yml", "")

        # Load Face Recognizer Model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(os.path.join(TRAINED_MODEL_DIR, model_file))
        recognizers[section_name] = recognizer

        # Load ID Mapping
        id_mapping_file = os.path.join(TRAINED_MODEL_DIR, f"{section_name}_id_mapping.txt")
        id_mappings[section_name] = {}

        if os.path.exists(id_mapping_file):
            with open(id_mapping_file, "r") as f:
                for line in f:
                    num_id, student_name = line.strip().split()
                    id_mappings[section_name][int(num_id)] = student_name

print("Models & ID mappings loaded!")

# Track attendance
attendance = {}

# Initialize Video Capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        detected_face = gray[y:y+h, x:x+w]

        best_match = None
        best_confidence = 100  # Lower is better for LBPH

        for section, recognizer in recognizers.items():
            label, confidence = recognizer.predict(detected_face)

            if confidence < best_confidence:
                best_match = (section, label, confidence)

        if best_match:
            section, label, confidence = best_match

            if confidence < 70:  # Match above 70%
                student_name = id_mappings[section].get(label, "Unknown")

                # Display Name & Section on separate lines
                cv2.putText(frame, student_name, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, section, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Mark attendance
                if student_name not in attendance:
                    attendance[student_name] = {"section": section, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Face Recognition & Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save Attendance to CSV
with open(ATTENDANCE_FILE, "w", newline="") as csvfile:
    fieldnames = ["Student Name", "Section", "Time"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for student_name, data in attendance.items():
        writer.writerow({"Student Name": student_name, "Section": data["section"], "Time": data["time"]})

print(f"Attendance saved to {ATTENDANCE_FILE}")

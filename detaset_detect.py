import cv2
import numpy as np
import os

# Path to trained models
MODEL_DIR = "trained_models"
DATASET_DIR = "dataset"

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load trained models based on sections
recognizers = {}
for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith(".yml"):
        section = model_file.replace("face_model_", "").replace(".yml", "")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(os.path.join(MODEL_DIR, model_file))
        recognizers[section] = recognizer

# Mapping IDs to names
student_names = {}
for student_dir in os.listdir(DATASET_DIR):
    parts = student_dir.split('_')
    if len(parts) >= 3:
        student_id = int(parts[0])
        name = parts[1]
        student_names[student_id] = name

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        predicted_id = None
        confidence_score = None
        predicted_name = "Unknown"

        # Try to recognize face using all section models
        for section, recognizer in recognizers.items():
            label, confidence = recognizer.predict(face_roi)
            if confidence < 100:  # Lower confidence is better
                predicted_id = label
                confidence_score = confidence
                predicted_name = student_names.get(label, "Unknown")

        # Display results
        color = (0, 255, 0) if predicted_name != "Unknown" else (0, 0, 255)
        text = f"{predicted_name} ({confidence_score:.2f})"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

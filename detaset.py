import cv2
import os
import requests

# API URL to fetch student details
API_URL = "http://localhost:8000/students/get_all_students"

# Fetch student details
response = requests.get(API_URL)
if response.status_code != 200:
    print("Failed to fetch student data")
    exit()

students = response.json().get("data", [])

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create dataset directory if not exists
DATASET_DIR = "dataset"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Load face detection model (Haarcascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for student in students:
    student_id = student["student_id"]
    name = student["name"].replace(" ", "_")  # Replace spaces with underscores
    standard = student["standard"]["grade"] + "_" + student["standard"]["section"]

    # Create directory for each student
    student_dir = os.path.join(DATASET_DIR, f"{student_id}_{name}_{standard}")
    if not os.path.exists(student_dir):
        os.makedirs(student_dir)

    print(f"Capturing images for {name} (ID: {student_id}, Standard: {standard})")

    count = 0
    while count < 100:  # Capture 100 images per student
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]  # Extract face region
            image_path = os.path.join(student_dir, f"{count+1}.jpg")
            cv2.imwrite(image_path, face)  # Save grayscale face image
            count += 1
            print(f"Saved: {image_path}")

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} - {student_id}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Capturing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture interrupted by user.")
            break

    print(f"Captured {count} images for {name}.\n")

cap.release()
cv2.destroyAllWindows()

print("Dataset collection completed successfully!")

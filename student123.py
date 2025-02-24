import cv2
import os
import numpy as np
from PIL import Image
import pickle
from sqlalchemy.orm import Session
from app.models.attendance import Attendance
from app.models.student import Student
from app.databases import get_db
import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "Image_for_Training")
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer.yml")
FACE_CASCADE = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

def create_dataset(student_id: str):
    cam = cv2.VideoCapture(0)
    count = 0
    student_folder = os.path.join(IMAGE_DIR, student_id)
    os.makedirs(student_folder, exist_ok=True)
    
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{student_folder}/{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
        cv2.imshow("Capturing Faces", frame)
        if count >= 50:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    ids = []
    student_ids = []
    
    for student_id in os.listdir(IMAGE_DIR):
        student_path = os.path.join(IMAGE_DIR, student_id)
        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            gray_image = Image.open(img_path).convert('L')
            image_np = np.array(gray_image, 'uint8')
            
            faces.append(image_np)
            ids.append(int(student_id))
            student_ids.append(student_id)
    
    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_PATH)
    
    with open("student_ids.pkl", "wb") as f:
        pickle.dump(student_ids, f)

def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    
    with open("student_ids.pkl", "rb") as f:
        student_ids = pickle.load(f)
    
    cam = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(face_img)
            
            if conf < 50:
                student_id = student_ids[id_]
                mark_attendance(student_id)
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Recognizing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

def mark_attendance(student_id: str):
    db: Session = next(get_db())
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if student:
        new_attendance = Attendance(student_id=student_id, date=datetime.date.today())
        db.add(new_attendance)
        db.commit()
        print(f"Attendance marked for {student.name}")
    else:
        print("Student not found!")

import cv2
import numpy as np
import os
from PIL import Image

# Path to dataset
DATASET_DIR = "dataset"
MODEL_DIR = "trained_models"

# Create model directory if not exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to get images and labels
# def get_images_and_labels(path):
#     image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
#     face_samples = []
#     ids = []

#     for image_path in image_paths:
#         gray_image = Image.open(image_path).convert('L')  # Convert to grayscale
#         image_np = np.array(gray_image, 'uint8')

#         # Extract folder name (student identifier)
#         folder_name = os.path.basename(os.path.dirname(image_path))
#         parts = folder_name.split('_')

#         # Ensure folder follows expected format {id}_{name}_{section}
#         if len(parts) < 3 or not parts[0].isdigit():
#             print(f"Skipping invalid folder: {folder_name}")
#             continue

#         student_id = int(parts[0])  # Convert ID to integer

#         faces = face_cascade.detectMultiScale(image_np)
#         for (x, y, w, h) in faces:
#             face_samples.append(image_np[y:y+h, x:x+w])
#             ids.append(student_id)

#     return face_samples, np.array(ids)

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_images_and_labels(path):
    face_samples = []
    ids = []

    for student_folder in os.listdir(path):
        student_path = os.path.join(path, student_folder)
        
        if not os.path.isdir(student_path):
            continue  # Skip non-folder files

        # Extract correct folder naming format
        parts = student_folder.split('_')

        # Ensure at least {uuid, name, standard, section}
        if len(parts) < 4:
            print(f"Skipping invalid folder: {student_folder}")
            continue
        
        # Extract student ID properly
        student_id = parts[0]  # UUID instead of integer
        name = parts[1]         # Student Name
        standard = parts[2]     # Standard
        section = parts[3]      # Section

        # Make sure student_id is valid (not empty)
        if not student_id:
            print(f"Skipping folder with invalid ID: {student_folder}")
            continue

        # Read images from the folder
        for image_file in os.listdir(student_path):
            if not image_file.endswith(".jpg"):
                continue  # Skip non-image files
            
            image_path = os.path.join(student_path, image_file)
            gray_image = Image.open(image_path).convert("L")  # Convert to grayscale
            image_np = np.array(gray_image, "uint8")

            # Detect faces
            faces = face_cascade.detectMultiScale(image_np)
            for (x, y, w, h) in faces:
                face_samples.append(image_np[y:y+h, x:x+w])
                ids.append(student_id)  # Keep as string instead of int

    return face_samples, np.array(ids, dtype=np.int32)




# # Train model for each section
# sections = set()
# for student_dir in os.listdir(DATASET_DIR):
#     parts = student_dir.split('_')
#     if len(parts) >= 3:
#         section = parts[2]  # Extract section (Grade_Section)
#         sections.add(section)

# for section in sections:
#     print(f"Training model for section: {section}")

#     # Collect images for this section
#     section_path = os.path.join(DATASET_DIR)
#     face_samples, ids = [], []

#     for student_dir in os.listdir(DATASET_DIR):
#         if section in student_dir:
#             student_path = os.path.join(DATASET_DIR, student_dir)
#             s_faces, s_ids = get_images_and_labels(student_path)
#             face_samples.extend(s_faces)
#             ids.extend(s_ids)

#     # Train the recognizer
#     recognizer.train(face_samples, np.array(ids))

#     # Save model
#     model_path = os.path.join(MODEL_DIR, f"face_model_{section}.yml")
#     recognizer.save(model_path)
#     print(f"Model saved: {model_path}")

# print("Training completed successfully!")
def train_model_for_section(section):
    section_path = os.path.join(DATASET_DIR, section)

    if not os.path.exists(section_path):
        print(f"Skipping missing section: {section}")
        return
    
    face_samples, ids = get_images_and_labels(section_path)

    if len(face_samples) == 0:
        print(f"Skipping training for {section} due to no valid data.")
        return

    print(f"Training model for section: {section}")

    # Train model
    recognizer.train(face_samples, ids)

    # Save the trained model
    model_filename = f"{section}.yml"
    recognizer.save(model_filename)
    print(f"Model saved as {model_filename}")

# Train models for all sections
sections = [folder for folder in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, folder))]
for section in sections:
    train_model_for_section(section)

print("Training completed!")
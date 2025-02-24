# FaceRecognitionModule
Mode TO Detect face and give unique id to backend


Project Description: Face Recognition Student Attendance Management System
Overview
The Face Recognition Student Attendance Management System is an innovative application designed to streamline and automate the process of taking student attendance in educational institutions. Utilizing advanced computer vision techniques and image processing, this system offers a reliable and efficient method for managing attendance records. The application is divided into two main flows: Admin and Student. The Admin has full control over the system, including managing student records and viewing detailed attendance reports, while the Student can only view their individual attendance report.

Key Features
Face Recognition: Employs the Haar Cascade Classifier for frontal face detection, ensuring accurate identification of students.
Image Processing: Utilizes histogram analysis to enhance image processing and face detection accuracy.
Admin and Student Flows: Distinct user roles with specific functionalities to ensure security and efficiency.
Admin Flow: Allows full control over the system including adding new students, updating records, and viewing comprehensive attendance reports.
Student Flow: Enables students to securely view their own attendance records.
Automated Attendance Marking: Automatically marks attendance based on face recognition, reducing the manual effort and errors associated with traditional methods.
Technology Stack
Programming Language: Python
Libraries: OpenCV (for computer vision), NumPy (for numerical operations), Pandas (for data manipulation)
Database: SQLite (for storing student records and attendance data)
GUI: Tkinter (for creating a user-friendly interface)
System Architecture
Face Detection and Recognition:

The system captures images from a camera feed.
Uses the Haar Cascade Classifier to detect faces in the frame.
Recognizes faces by comparing with stored face data using histogram analysis for improved accuracy.
Database Management:

Stores student details and attendance records in an SQLite database.
Admin can add, update, or delete student records.
Attendance Marking:

On recognizing a face, the system automatically marks the student as present.
Stores the date and time of attendance in the database.
User Interface:

Admin can log in to manage the system, view detailed reports, and modify student data.
Students can log in to view their own attendance records.
Detailed Description of Functionalities
Admin Flow:
Login: Admin logs into the system using a secure login form.
Dashboard: Displays an overview of attendance statistics and student records.
Manage Students: Admin can add new students by entering their details and capturing their face data. They can also update or delete existing student records.
View Reports: Admin can view detailed attendance reports, including daily, weekly, and monthly summaries.
Settings: Allows the admin to configure system settings, such as camera options and recognition thresholds.
Student Flow:
Login: Students log into the system using their unique credentials.
View Attendance: Students can view their attendance records, including dates and times they were marked present.
Image Examples
Face Detection Using Haar Cascade:
Face Detection

Histogram Analysis for Image Processing:
Histogram Analysis

Implementation Steps
Setup and Installation:

Install necessary libraries:
bash
Copy code
pip install opencv-python numpy pandas
Set up the SQLite database and create tables for storing student and attendance records.
Face Detection Module:

Load the Haar Cascade model:

python
Copy code
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Capture images and detect faces:

python
Copy code
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces
Histogram Analysis for Improved Recognition:

Convert image to grayscale and compute histogram:
python
Copy code
def compute_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist
Database Operations:

Connect to SQLite database and create necessary tables:
python
Copy code
import sqlite3
conn = sqlite3.connect('attendance.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT, face_data BLOB)''')
c.execute('''CREATE TABLE IF NOT EXISTS attendance (student_id INTEGER, date TEXT, time TEXT)''')
conn.commit()
GUI Design:

Use Tkinter to create login forms, dashboards, and management interfaces for both Admin and Student flows.
By following the above implementation steps and using the provided images as references, you can build a robust and efficient Face Recognition Student Attendance Management System. This project not only automates attendance marking but also ensures accuracy and security in managing student attendance records.





![Screenshot 2024-05-22 160912](https://github.com/satyam00043/facerecognition/assets/114933291/4511d5c9-a3a4-496a-8e5f-8a2ceb244ea0)

Our system u![Screenshot 2024-05-22 161001](https://github.com/satyam00043/facerecognition/assets/114933291/421d1117-33c0-4b60-b59c-4572c5c0905f)
tilizes Tkinter for the user interface, offering a seamless interaction experience. Leveraging OpenCV, we employ cutting-edge image detection and recognition techniques to identify faces accurately. Through the integration of MySQL, we establish a robust database to store and manage attendance records efficiently.
![Screenshot 2024-05-22 161033](https://github.com/satyam00043/facerecognition/assets/114933291/461d6d5d-a01d-44e7-858b-8e9ab362f6d4)

Upon launching the application, users are greeted with a user-friendly interface where they can perform various actions such as registering new faces, taking attendance, and viewing attendance reports. The face recognition module employs advanced algorithms to match faces against stored images in the database, ensuring high precision in identification.

When a user requests attendance, the system captures live images, processes them through OpenCV for face detection and recognition, and then logs the attendance status in the MySQL database. Administrators can easily track attendance records, generate reports, and manage user profiles through the intuitive interface.

Our solution not only streamlines the attendance tracking process but also enhances security and accuracy through advanced face recognition technology. With its user-friendly interface and robust backend, our attendance management system offers a comprehensive solution for organizations of all sizes.


![Screenshot 2024-05-14 153530](https://github.com/satyam00043/facerecognition/assets/114933291/beef1a7c-158c-4780-8c99-bdac523843b6)
![Screenshot 2024-05-14 153552](https://github.com/satyam00043/facerecognition/assets/114933291/e2e354aa-f3b8-4b25-a5c9-49f5e6d65b34)
![Screenshot 2024-05-14 153643](https://github.com/satyam00043/facerecognition/assets/114933291/9dfa325d-957b-49a3-871f-7ac622f729a4)

![Screenshot 2024-05-14 153821]![Screenshot 2024-05-14 154230](https://github.com/satyam00043/facerecognition/assets/114933291/405fc202-7611-4eb0-a852-982c5cc96e9c)
(https://github.com/satyam00043/facerecognition/assets/114933291/b3f57e84-27fe-4fac-80fc-0ae1cf532851)![Screenshot 2024-05-14 160103](https://github.com/satyam00043/facerecognition/assets/114933291/58606e27-a4a3-4c02-b537-839b9a24a067)

![Screenshot 2024-05-14 155424](https://github.com/satyam00043/facerecognition/assets/114933291/036d83c6-a900-49c1-a869-fef4e4a9f867)



![Screenshot 2024-05-14 160103](https://github.com/satyam00043/facerecognition/assets/114933291/7d14f58c-b01d-4332-acd2-38702828a7be)



Student View  

![Screenshot 2024-06-03 170946](https://github.com/satyam00043/facerecognition/assets/114933291/230c67c3-6ab7-48c0-b2e5-cfe9b129c288)





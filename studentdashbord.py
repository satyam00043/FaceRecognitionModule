import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd


class StudentDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Dashboard")
        self.root.geometry("1530x800")
        self.root.configure(bg="light blue")

        self.label_profile = tk.Label(self.root, text="Student Profile", font=("Arial", 14, "bold"), bg="light blue")
        self.label_profile.pack(pady=10)

        # Create a frame for profile information
        self.frame_profile = tk.Frame(self.root, bg="light blue")
        self.frame_profile.pack()

        # Student information labels
        self.label_name = tk.Label(self.frame_profile, text="Name:", font=("Arial", 12), bg="light blue")
        self.label_name.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.label_class = tk.Label(self.frame_profile, text="Class:", font=("Arial", 12), bg="light blue")
        self.label_class.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.label_teacher = tk.Label(self.frame_profile, text="Class Teacher:", font=("Arial", 12), bg="light blue")
        self.label_teacher.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        # Sample student information (replace with actual data)
        self.student_info = {
            "name": "Satyam",
            "class": "CSE",
            "class_teacher": "Mr. Smith"
        }

        # Display student information
        self.display_profile_info()

        # Attendance section
        self.label_attendance = tk.Label(self.root, text="Attendance", font=("Arial", 14, "bold"), bg="light blue")
        self.label_attendance.pack(pady=10)

        # Create a frame for attendance
        self.frame_attendance = tk.Frame(self.root, bg="light blue")
        self.frame_attendance.pack()

        # Sample attendance data (replace with actual data)
        self.attendance_data = {
            "Class": ["Math", "Science", "English"],
            "Attendance": ["Present", "Absent", "Present"],
            "Teacher": ["Ms. Johnson", "Mr. Brown", "Ms. Adams"]
        }

        # Display attendance
        self.display_attendance()

        # Attendance analysis button
        self.button_analysis = tk.Button(self.root, text="View Attendance Analysis", font=("Arial", 12),
                                          command=self.view_analysis)
        self.button_analysis.pack(pady=10)

    def display_profile_info(self):
        # Display student information
        name = self.student_info["name"]
        class_name = self.student_info["class"]
        teacher = self.student_info["class_teacher"]

        self.label_name.config(text=f"Name: {name}")
        self.label_class.config(text=f"Class: {class_name}")
        self.label_teacher.config(text=f"Class Teacher: {teacher}")

    def display_attendance(self):
        # Display attendance data in a table
        tree = ttk.Treeview(self.frame_attendance, columns=("Class", "Attendance", "Teacher"), show="headings")
        tree.heading("Class", text="Class")
        tree.heading("Attendance", text="Attendance")
        tree.heading("Teacher", text="Teacher")
        tree.pack()

        for i in range(len(self.attendance_data["Class"])):
            tree.insert("", "end", values=(self.attendance_data["Class"][i],
                                           self.attendance_data["Attendance"][i],
                                           self.attendance_data["Teacher"][i]))

    def view_analysis(self):
        # Add code to display attendance analysis
        messagebox.showinfo("Attendance Analysis", "Feature under development")


if __name__ == "__main__":
    root = tk.Tk()
    app = StudentDashboard(root)
    root.mainloop()
# import tkinter as tk
# from tkinter import ttk
# from tkinter import messagebox
# import pandas as pd
# import mysql.connector

# class StudentDashboard:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Student Dashboard")
#         self.root.geometry("1530x800")
#         self.root.configure(bg="light blue")

#         # Database connection details
#         self.db_connection = mysql.connector.connect(
#             host="localhost",
#             user="root",
#             password="satyam",
#             database="facerecognition"
#         )
#         self.db_cursor = self.db_connection.cursor(dictionary=True)

#         self.logged_in_user = self.get_logged_in_user()

#         self.label_profile = tk.Label(self.root, text="Student Profile", font=("Arial", 14, "bold"), bg="light blue")
#         self.label_profile.pack(pady=10)

#         # Create a frame for profile information
#         self.frame_profile = tk.Frame(self.root, bg="light blue")
#         self.frame_profile.pack()

#         # Student information labels
#         self.label_name = tk.Label(self.frame_profile, text="Name:", font=("Arial", 12), bg="light blue")
#         self.label_name.grid(row=0, column=0, padx=5, pady=5, sticky="w")
#         self.label_class = tk.Label(self.frame_profile, text="Class:", font=("Arial", 12), bg="light blue")
#         self.label_class.grid(row=1, column=0, padx=5, pady=5, sticky="w")
#         self.label_teacher = tk.Label(self.frame_profile, text="Class Teacher:", font=("Arial", 12), bg="light blue")
#         self.label_teacher.grid(row=2, column=0, padx=5, pady=5, sticky="w")

#         if self.logged_in_user:
#             self.student_info = {
#                 "name": self.logged_in_user["name"],
#                 "class": self.logged_in_user["class"],
#                 "class_teacher": self.logged_in_user["class_teacher"]
#             }

#             # Display student information
#             self.display_profile_info()

#             # Attendance section
#             self.label_attendance = tk.Label(self.root, text="Attendance", font=("Arial", 14, "bold"), bg="light blue")
#             self.label_attendance.pack(pady=10)

#             # Create a frame for attendance
#             self.frame_attendance = tk.Frame(self.root, bg="light blue")
#             self.frame_attendance.pack()

#             # Load and display attendance data
#             self.attendance_data = self.load_attendance_data('atte.csv')
#             self.display_attendance()

#             # Attendance analysis button
#             self.button_analysis = tk.Button(self.root, text="View Attendance Analysis", font=("Arial", 12),
#                                               command=self.view_analysis)
#             self.button_analysis.pack(pady=10)
#         else:
#             messagebox.showerror("Error", "User not found or not logged in")

#     def get_logged_in_user(self):
#         # This function should return the details of the logged-in user.
#         # For demonstration purposes, we are fetching the first user from the users table.
#         # You should replace this with the actual logged-in user logic.
#         self.db_cursor.execute("SELECT name FROM users LIMIT 1")
#         user = self.db_cursor.fetchone()
#         return user

#     def display_profile_info(self):
#         # Display student information
#         name = self.student_info["name"]
#         class_name = self.student_info["class"]
#         teacher = self.student_info["class_teacher"]

#         self.label_name.config(text=f"Name: {name}")
#         self.label_class.config(text=f"Class: {class_name}")
#         self.label_teacher.config(text=f"Class Teacher: {teacher}")

#     def load_attendance_data(self, file_path):
#         # Read data from the CSV file
#         df = pd.read_csv(file_path)
#         student_name = self.student_info['name']

#         # Filter the data for the specific student
#         student_data = df[df['Name'] == student_name]

#         # Convert the DataFrame to a dictionary
#         attendance_data = student_data.to_dict(orient='list')
#         return attendance_data

#     def display_attendance(self):
#         # Display attendance data in a table
#         tree = ttk.Treeview(self.frame_attendance, columns=("Class", "Attendance", "Teacher"), show="headings")
#         tree.heading("Class", text="Class")
#         tree.heading("Attendance", text="Attendance")
#         tree.heading("Teacher", text="Teacher")
#         tree.pack()

#         for i in range(len(self.attendance_data["Class"])):
#             tree.insert("", "end", values=(self.attendance_data["Class"][i],
#                                            self.attendance_data["Attendance"][i],
#                                            self.attendance_data["Teacher"][i]))

#     def view_analysis(self):
#         # Perform attendance analysis
#         total_classes = len(self.attendance_data["Attendance"])
#         present_count = self.attendance_data["Attendance"].count("Present")
#         attendance_percentage = (present_count / total_classes) * 100

#         analysis_message = f"Total Classes: {total_classes}\nPresent: {present_count}\nAttendance Percentage: {attendance_percentage:.2f}%"
#         messagebox.showinfo("Attendance Analysis", analysis_message)


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = StudentDashboard(root)
# #     root.mainloop()
# import tkinter as tk
# from tkinter import ttk
# from tkinter import messagebox
# import pandas as pd
# import mysql.connector

# class StudentDashboard:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Student Dashboard")
#         self.root.geometry("1530x800")
#         self.root.configure(bg="light blue")

#         # Database connection details
#         try:
#             self.db_connection = mysql.connector.connect(
#                 host="localhost",
#                 user="root",
#                 password="satyam",
#                 database="facerecognition"
#             )
#             self.db_cursor = self.db_connection.cursor(dictionary=True)
#         except mysql.connector.Error as err:
#             messagebox.showerror("Database Error", f"Error connecting to the database: {err}")
#             self.root.destroy()
#             return

#         self.logged_in_user = self.get_logged_in_user()

#         self.label_profile = tk.Label(self.root, text="Student Profile", font=("Arial", 14, "bold"), bg="light blue")
#         self.label_profile.pack(pady=10)

#         # Create a frame for profile information
#         self.frame_profile = tk.Frame(self.root, bg="light blue")
#         self.frame_profile.pack()

#         # Student information labels
#         self.label_name = tk.Label(self.frame_profile, text="Name:", font=("Arial", 12), bg="light blue")
#         self.label_name.grid(row=0, column=0, padx=5, pady=5, sticky="w")
#         self.label_class = tk.Label(self.frame_profile, text="Class:", font=("Arial", 12), bg="light blue")
#         self.label_class.grid(row=1, column=0, padx=5, pady=5, sticky="w")
#         self.label_teacher = tk.Label(self.frame_profile, text="Class Teacher:", font=("Arial", 12), bg="light blue")
#         self.label_teacher.grid(row=2, column=0, padx=5, pady=5, sticky="w")

#         if self.logged_in_user:
#             self.student_info = {
#                 "name": self.logged_in_user["name"],
#                 "class": self.logged_in_user["class"],
#                 "class_teacher": self.logged_in_user["class_teacher"]
#             }

#             # Display student information
#             self.display_profile_info()

#             # Attendance section
#             self.label_attendance = tk.Label(self.root, text="Attendance", font=("Arial", 14, "bold"), bg="light blue")
#             self.label_attendance.pack(pady=10)

#             # Create a frame for attendance
#             self.frame_attendance = tk.Frame(self.root, bg="light blue")
#             self.frame_attendance.pack()

#             # Load and display attendance data
#             self.attendance_data = self.load_attendance_data('atte.csv')
#             self.display_attendance()

#             # Attendance analysis button
#             self.button_analysis = tk.Button(self.root, text="View Attendance Analysis", font=("Arial", 12),
#                                              command=self.view_analysis)
#             self.button_analysis.pack(pady=10)
#         else:
#             messagebox.showerror("Error", "User not found or not logged in")
#             self.root.destroy()

#     def get_logged_in_user(self):
#         # This function should return the details of the logged-in user.
#         # For demonstration purposes, we are fetching the first user from the users table.
#         # You should replace this with the actual logged-in user logic.
#         self.db_cursor.execute("SELECT name, class, class_teacher FROM users LIMIT 1")
#         user = self.db_cursor.fetchone()
#         return user

#     def display_profile_info(self):
#         # Display student information
#         name = self.student_info["name"]
#         class_name = self.student_info["class"]
#         teacher = self.student_info["class_teacher"]

#         self.label_name.config(text=f"Name: {name}")
#         self.label_class.config(text=f"Class: {class_name}")
#         self.label_teacher.config(text=f"Class Teacher: {teacher}")

#     def load_attendance_data(self, file_path):
#         # Read data from the CSV file
#         df = pd.read_csv(file_path)
#         student_name = self.student_info['name']

#         # Filter the data for the specific student
#         student_data = df[df['Name'] == student_name]

#         # Convert the DataFrame to a dictionary
#         attendance_data = student_data.to_dict(orient='list')
#         return attendance_data

#     def display_attendance(self):
#         # Display attendance data in a table
#         tree = ttk.Treeview(self.frame_attendance, columns=("Date", "Attendance"), show="headings")
#         tree.heading("Date", text="Date")
#         tree.heading("Attendance", text="Attendance")
#         tree.pack()

#         for i in range(len(self.attendance_data["Date"])):
#             tree.insert("", "end", values=(self.attendance_data["Date"][i],
#                                            self.attendance_data["Attendance"][i]))

#     def view_analysis(self):
#         # Perform attendance analysis
#         total_classes = len(self.attendance_data["Attendance"])
#         present_count = self.attendance_data["Attendance"].count("Present")
#         attendance_percentage = (present_count / total_classes) * 100

#         analysis_message = f"Total Classes: {total_classes}\nPresent: {present_count}\nAttendance Percentage: {attendance_percentage:.2f}%"
#         messagebox.showinfo("Attendance Analysis", analysis_message)


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = StudentDashboard(root)
#     root.mainloop()


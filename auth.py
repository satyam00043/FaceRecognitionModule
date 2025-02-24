from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import mysql.connector
import bcrypt
from main2 import Face_recognition_system
from studentdashbord import StudentDashboard
from PIL import Image,ImageTk

def main():
    win = Tk()
    app = Login_Window(win)
    win.mainloop()

class Login_Window:
    def __init__(self, root):
        self.root = root
      
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System College Project")
        # Load the image
        img = Image.open("C:/Users/satya/Desktop/Examprep/machine/project2final/image/face10.jpeg")
        
        # Resize the image
        img = img.resize((1510, 890), Image.BILINEAR)  # Use BILINEAR as an example
         # Convert the resized image to PhotoImage
        self.photoimg = ImageTk.PhotoImage(img)
        # Create label and place the image
        f_lbl = Label(self.root, image=self.photoimg)
        f_lbl.place(x=0, y=0, width=1530, height=800)
        

        # frame = Frame(self.root,bg="white", highlightthickness=0)
        # frame.place(x=300, y=200, width=450, height=500)
        
        get_str = Label(self.root, text="Get Started", font=("times new roman", 20, "bold"), fg="white", bg="black")
        get_str.place(x=650, y=280)

        User_str = Label(self.root, text="User_Name", font=("times new roman", 20, "bold"), fg="white", bg="black")
        User_str.place(x=590, y=325)

        self.txtuser = StringVar()
        self.txtpass = StringVar()

        txtuser = ttk.Entry(self.root, textvariable=self.txtuser, font=("times new roman", 15, "bold"))
        txtuser.place(x=590, y=360, width=300)

        password = Label(self.root, text="Password", font=("times new roman", 20, "bold"), fg="white", bg="black")
        password.place(x=590, y=395)

        txtpass = ttk.Entry(self.root, textvariable=self.txtpass, font=("times new roman", 15, "bold"), show="*")
        txtpass.place(x=590, y=430, width=300)

        btn_login = Button(self.root, text="Login", font=("times new roman", 20, "bold"), fg="white",bg="gray", command=self.login)
        btn_login.place(x=590, y=500, width=150)

        # btn_register = Button(self.root, text="Register", font=("times new roman", 25, "bold"),fg="blue",bg="gray" ,command=self.open_register_window)
        # btn_register.place(x=750, y=500, width=150, height=50)

        btn_forgot = Button(self.root, text="Forgot Password?", font=("times new roman", 15, "bold"), fg="white" ,bg="gray", command=self.open_forgot_password_window)
        btn_forgot.place(x=750, y=500, width=150 ,height=55)

    def login(self):
        username = self.txtuser.get()
        password = self.txtpass.get()
        if authenticate(username, password)==1:
            self.open_face_recognition_system()
        elif authenticate(username, password)==2:
            self.open_student_dashboard()
        else:
            "u r not registered here"

    def open_register_window(self):
        self.new_window = Toplevel(self.root)
        self.app = Register_Window(self.new_window)

    def open_forgot_password_window(self):
        self.new_window = Toplevel(self.root)
        self.app = Forgot_Password_Window(self.new_window)

    def open_face_recognition_system(self):
        self.new_window = Toplevel(self.root)
        self.app = Face_recognition_system(self.new_window)
    def open_student_dashboard(self):
        self.new_window = Toplevel(self.root)
        self.app = StudentDashboard(self.new_window)

class Register_Window:
    def __init__(self, root):
        self.root = root
        self.root.title("Register")
        self.root.geometry("400x400+0+0")

        frame = Frame(self.root, bg="black")
        frame.place(x=50, y=50, width=300, height=300)

        register_str = Label(frame, text="Register", font=("times new roman", 20, "bold"), fg="white", bg="black")
        register_str.place(x=100, y=20)

        user_label = Label(frame, text="Username", font=("times new roman", 15, "bold"), fg="white", bg="black")
        user_label.place(x=30, y=70)
        self.reg_user = StringVar()
        reg_user_entry = ttk.Entry(frame, textvariable=self.reg_user, font=("times new roman", 15, "bold"))
        reg_user_entry.place(x=30, y=100, width=200)

        pass_label = Label(frame, text="Password", font=("times new roman", 15, "bold"), fg="white", bg="black")
        pass_label.place(x=30, y=140)
        self.reg_pass = StringVar()
        reg_pass_entry = ttk.Entry(frame, textvariable=self.reg_pass, font=("times new roman", 15, "bold"), show="*")
        reg_pass_entry.place(x=30, y=170, width=200)

        btn_register = Button(frame, text="Register", font=("times new roman", 15, "bold"), command=self.register_user)
        btn_register.place(x=100, y=220, width=100)

    def register_user(self):
        username = self.reg_user.get()
        password = self.reg_pass.get()
        insert_user(username, password)

class Forgot_Password_Window:
    def __init__(self, root):
        self.root = root
        self.root.title("Reset Password")
        self.root.geometry("400x400+0+0")

        frame = Frame(self.root, bg="black")
        frame.place(x=50, y=50, width=300, height=300)

        reset_str = Label(frame, text="Reset Password", font=("times new roman", 20, "bold"), fg="white", bg="black")
        reset_str.place(x=50, y=20)

        user_label = Label(frame, text="Username", font=("times new roman", 15, "bold"), fg="white", bg="black")
        user_label.place(x=30, y=70)
        self.forgot_user = StringVar()
        forgot_user_entry = ttk.Entry(frame, textvariable=self.forgot_user, font=("times new roman", 15, "bold"))
        forgot_user_entry.place(x=30, y=100, width=200)

        pass_label = Label(frame, text="New Password", font=("times new roman", 15, "bold"), fg="white", bg="black")
        pass_label.place(x=30, y=140)
        self.forgot_pass = StringVar()
        forgot_pass_entry = ttk.Entry(frame, textvariable=self.forgot_pass, font=("times new roman", 15, "bold"), show="*")
        forgot_pass_entry.place(x=30, y=170, width=200)

        btn_reset = Button(frame, text="Reset", font=("times new roman", 15, "bold"), command=self.reset_password)
        btn_reset.place(x=100, y=220, width=100)

    def reset_password(self):
        username = self.forgot_user.get()
        new_password = self.forgot_pass.get()
        update_password(username, new_password)

def authenticate(username, password):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="satyam",
            database="facerecognition"
        )

        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        
        user = cursor.fetchone()

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            access = user['access']
            
            messagebox.showinfo("Login Successful", f"Welcome {username}!\nYour access: {access} this is your ide")
            if access==0:
                return 2
            elif access==1:
                return 1
            else:
                return 3
       
        
                
        else:
            messagebox.showerror("Login Failed", "Invalid username or password")
            return None
            

    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error: {err}")
        return False

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def insert_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="satyam",
            database="facerecognition"
        )

        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO users (username, password, access) VALUES (%s, %s, %s)",
            (username, hashed_password.decode('utf-8'), 'default access')
        )
        connection.commit()
        messagebox.showinfo("Success", "Registration Successful!")

    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error: {err}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def update_password(username, new_password):
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="satyam",
            database="facerecognition"
        )

        cursor = connection.cursor()
        cursor.execute(
            "UPDATE users SET password = %s WHERE username = %s",
            (hashed_password.decode('utf-8'), username)
        )
        connection.commit()
        
        if cursor.rowcount == 0:
            messagebox.showerror("Error", "Username not found")
        else:
            messagebox.showinfo("Success", "Password Reset Successful!")

    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error: {err}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    main()

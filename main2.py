from tkinter import *
from PIL import Image, ImageTk
from student import Student
import os
from time import strftime
from datetime import datetime
from train import Train
from face_recognition import Face_Recognition
from exit import Exit
from developer import Developer
from attendence import Attendence
from help import Help


class Face_recognition_system:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x1000")  # Adjusted window size for better visibility
        self.root.title("Face Recognition Attendance System")

        # Load and resize the first image
        img_path1 = "image/face1.jpg"
        img1 = Image.open(img_path1)
        img1 = img1.resize((510, 190), Image.BILINEAR)
        self.photoimg1 = ImageTk.PhotoImage(img1)
        f_lbl1 = Label(self.root, image=self.photoimg1)
        f_lbl1.place(x=0, y=0)

        # Load and resize the second image
        img_path2 = "image/face.jpg"
        img2 = Image.open(img_path2)
        img2 = img2.resize((510, 190), Image.BILINEAR)
        self.photoimg2 = ImageTk.PhotoImage(img2)
        f_lbl2 = Label(self.root, image=self.photoimg2)
        f_lbl2.place(x=540, y=0)

        # Load and resize the third image
        img_path3 = "image/face.jpg"
        img3 = Image.open(img_path3)
        img3 = img3.resize((510, 190), Image.BILINEAR)
        self.photoimg3 = ImageTk.PhotoImage(img3)
        f_lbl3 = Label(self.root, image=self.photoimg3)
        f_lbl3.place(x=1080, y=0)

        # Load and resize the background image
        bg_img_path = "image/face6.jpg"
        bg_img = Image.open(bg_img_path)
        bg_img = bg_img.resize((1530, 790), Image.BILINEAR)
        self.photoimg4 = ImageTk.PhotoImage(bg_img)
        bg_lbl = Label(self.root, image=self.photoimg4)
        bg_lbl.place(x=0, y=210)

        # Title label
        title_lbl = Label(self.root, text="FACE RECOGNITION ATTENDANCE SYSTEM", font=("times new roman", 45, "bold"), bg="white", fg="blue")
        title_lbl.place(x=0, y=200, width=1400, height=45)
        #def time
        def time():
            string = strftime("%H:%M:%S %p")
            lbl1.config(text=string)
            lbl1.after(1000, time)
        lbl1=Label(f_lbl1,font=('times new roman',14,'bold'),background='white',fg="blue")
        lbl1.place(x=0,y=0,width=110,height=40)
        time()
        def date():
            now=datetime.now()
            d1 = now.strftime("%d/%m/%Y")
            lbl12.config(text=d1)
            lbl12.after(1000, time)
        lbl12=Label(f_lbl1,font=('times new roman',14,'bold'),background='white',fg="blue")
        lbl12.place(x=0,y=50,width=110,height=40)
        date()

        # Load and resize the button image
        button_img_path = "image/details.png"
        button_img = Image.open(button_img_path)
        button_img = button_img.resize((190, 190), Image.BILINEAR)
        self.photoimg5 = ImageTk.PhotoImage(button_img)


        # Create buttons with individual images
        b1 = Button(self.root, image=self.photoimg5,command=self.student_details, cursor="hand2")
        b1.place(x=100, y=300, width=200, height=220)
        b11 = Button(self.root, text="Student Details",command=self.student_details, cursor="hand2" ,font=("times new roman", 15, "bold"), bg="white", fg="blue")
        b11.place(x=100, y=500, width=200, height=40)

        #button photos
        button_img_path8 = "image/OIP.jpeg"
        button_img8 = Image.open(button_img_path8)
        button_img8 = button_img8.resize((190, 190), Image.BILINEAR)
        self.photoimg8 = ImageTk.PhotoImage(button_img8)

        b12 = Button(self.root, image=self.photoimg8,command=self.open_data ,cursor="hand2")
        b12.place(x=380, y=300, width=200, height=200)
        b112 = Button(self.root, text="Photos",command=self.open_data ,cursor="hand2" ,font=("times new roman", 15, "bold"), bg="white", fg="blue")
        b112.place(x=380, y=500, width=200, height=40)
        #button face detection
        button_img_path13 = "image/face1.jpg"
        button_img13 = Image.open(button_img_path13)
        button_img13 = button_img13.resize((190, 190), Image.BILINEAR)
        self.photoimg13 = ImageTk.PhotoImage(button_img13)

        b2 = Button(self.root, image=self.photoimg13,command=self.face_recoginitionn, cursor="hand2")
        b2.place(x=700, y=300, width=200, height=200)
        b21 = Button(self.root, text="face detection",command=self.face_recoginitionn ,cursor="hand2" ,font=("times new roman", 15, "bold"), bg="white", fg="blue")
        b21.place(x=700, y=500, width=200, height=40)

        button_img_path4 = "image/attendence.png"
        button_img9 = Image.open(button_img_path4)
        button_img9 = button_img9.resize((190, 190), Image.BILINEAR)
        self.photoimg9= ImageTk.PhotoImage(button_img9)

        b3 = Button(self.root, image=self.photoimg9,command=self.attendence_user, cursor="hand2")
        b3.place(x=1000, y=300, width=200, height=200)
        b31 = Button(self.root, text="Attendence", cursor="hand2" ,command=self.attendence_user,font=("times new roman", 15, "bold"), bg="white", fg="blue")
        b31.place(x=1000, y=500, width=200, height=40)
   
        #2nd line
        button_img_path10 = "image/train.jpg"
        button_img10 = Image.open(button_img_path10)
        button_img10 = button_img10.resize((190, 190), Image.BILINEAR)
        self.photoimg10 = ImageTk.PhotoImage(button_img10)

        b2 = Button(self.root, image=self.photoimg10,command=self.train_dataset, cursor="hand2")
        b2.place(x=380, y=550, width=200, height=200)
        b21 = Button(self.root, text="Train Data",command=self.train_dataset, cursor="hand2" ,font=("times new roman", 15, "bold"), bg="white", fg="blue")
        b21.place(x=380, y=750, width=200, height=40)

        button_img_path3 = "image/dev.jpg"
        button_img3 = Image.open(button_img_path3)
        button_img3= button_img3.resize((190, 190), Image.BILINEAR)
        self.photoimg7 = ImageTk.PhotoImage(button_img3)


        b3 = Button(self.root, image=self.photoimg7,command=self.developer_list ,cursor="hand2")
        b3.place(x=700, y=550, width=200, height=200)
        b31 = Button(self.root, text="Developer Team", cursor="hand2",command=self.developer_list ,font=("times new roman", 15, "bold"), bg="white", fg="blue")
        b31.place(x=700, y=750, width=200, height=40)

        button_img_path2 = "image/Helpdesk.png"
        button_img1 = Image.open(button_img_path2)
        button_img1 = button_img1.resize((190, 190), Image.BILINEAR)
        self.photoimg6 = ImageTk.PhotoImage(button_img1)


        b41 = Button(self.root, image=self.photoimg6,command=self.help_user, cursor="hand2")
        b41.place(x=1000, y=550, width=200, height=200)
        b411 = Button(self.root, text="help", cursor="hand2",command=self.help_user ,font=("times new roman", 15, "bold"), bg="white", fg="blue")
        b411.place(x=1000, y=750, width=200, height=40)
         #button exit
        button_img_path12 = "image/exit.jpeg"
        button_img12 = Image.open(button_img_path12)
        button_img12 = button_img12.resize((190, 190), Image.BILINEAR)
        self.photoimg12 = ImageTk.PhotoImage(button_img12)
        b4 = Button(self.root, image=self.photoimg12,command=self.exit_user, cursor="hand2")
        b4.place(x=1300, y=300, width=200, height=200)
        b41 = Button(self.root, text="Exit", cursor="hand2" ,command=self.exit_user,font=("times new roman", 15, "bold"), bg="white", fg="blue")
        b41.place(x=1300, y=500, width=200, height=40)




    def open_data(self):
         os.startfile("data")


    def  student_details(self):
            self.new_window=Toplevel(self.root)
            self.app=Student(self.new_window)
    def  train_dataset(self):
            self.new_window=Toplevel(self.root)
            self.app=Train(self.new_window)
    def face_recoginitionn(self):
          self.new_window=Toplevel(self.root)
          self.app=Face_Recognition(self.new_window)
    def help_user(self):
          self.new_window=Toplevel(self.root)
          self.app=Help(self.new_window)
    def exit_user(self):
          self.new_window=Toplevel(self.root)
          self.app=Exit(self.new_window)
    def attendence_user(self):
          self.new_window=Toplevel(self.root)
          self.app=Attendence(self.new_window)
    def developer_list(self):
          self.new_window=Toplevel(self.root)
          self.app=Developer(self.new_window)
    



















if __name__ == "__main__":
    root = Tk()
    obj = Face_recognition_system(root)
    root.mainloop()

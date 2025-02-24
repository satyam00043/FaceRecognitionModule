from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import mysql.connector
import cv2
import os
import numpy as np

class Help:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")
        title_lbl = Label(self.root, text="Face Recognition", font=("times new roman", 35, "bold"), bg="white", fg="darkgreen")
        title_lbl.place(x=0, y=0, width=1530, height=55)
         # Load and resize the first image
        img_path1 = "image/face.jpg"
        img1 = Image.open(img_path1)
        img1 = img1.resize((1510, 740), Image.BILINEAR)
        self.photoimg1 = ImageTk.PhotoImage(img1)
        f_lbl1 = Label(self.root, image=self.photoimg1)
        f_lbl1.place(x=0, y=55)

        b11 = Button(self.root, text="Face recognition", cursor="hand2" ,font=("times new roman", 20, "bold"), bg="blue", fg="white")
        b11.place(x=1110, y=550,width=250, height=55)


        # Load and resize the second image
        












if __name__ == "__main__":
    root = Tk()
    obj = Help(root)
    root.mainloop()

from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import mysql.connector
import cv2
import os
import numpy as np

class Train:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x890+0+0")
        self.root.title("Face Recognition System")

        title_lbl = Label(self.root, text="Train Dataset", font=("times new roman", 35, "bold"), bg="white", fg="darkgreen")
        title_lbl.place(x=0, y=0, width=1530, height=45)
         # Load and resize the first image
        img_path1 = "image/face.jpg"
        img1 = Image.open(img_path1)
        img1 = img1.resize((1510, 390), Image.BILINEAR)
        self.photoimg1 = ImageTk.PhotoImage(img1)
        f_lbl1 = Label(self.root, image=self.photoimg1)
        f_lbl1.place(x=0, y=45)

        # Load and resize the second image
        img_path2 = "image/face.jpg"
        img2 = Image.open(img_path2)
        img2 = img2.resize((1510, 390), Image.BILINEAR)
        self.photoimg2 = ImageTk.PhotoImage(img2)
        f_lbl2 = Label(self.root, image=self.photoimg2)
        f_lbl2.place(x=0, y=490)

        #button
        b11 = Button(self.root, text="Train Data",command=self.train_classifier, cursor="hand2" ,font=("times new roman", 30, "bold"), bg="lightgreen", fg="blue")
        b11.place(x=0, y=437,width=1510, height=55)

    def train_classifier(self):
        data_dir=("data")
        path=[os.path.join(data_dir,file) for file in os.listdir(data_dir)]

        faces=[]
        ids=[]
        for image in path:
            img=Image.open(image).convert('L')
            imageNp=np.array(img,'uint8')
            id=int(os.path.split(image)[1].split('.')[1])
            faces.append(imageNp)
            ids.append(id)
            cv2.imshow("Training",imageNp)
            cv2.waitKey(1)==13
        ids=np.array(ids)
        #Train the classifier and save
        clf=cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces,ids)
        clf.write("classifier.xml")
        cv2.destroyAllWindows()
        messagebox.showinfo("Result","Training datasets completed")



if __name__ == "__main__":
    root = Tk()
    obj = Train(root)
    root.mainloop()

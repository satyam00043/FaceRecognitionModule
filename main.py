
from tkinter import *
from PIL import Image, ImageTk
from student import Student
class Face_recognition_system:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System College Project")
        # Load the image
        img = Image.open("C:/Users/satya/Desktop/machine/project2final/image/face7.jpeg")
        # Resize the image
        img = img.resize((510, 190), Image.BILINEAR)  # Use BILINEAR as an example
         # Convert the resized image to PhotoImage
        self.photoimg = ImageTk.PhotoImage(img)
        # Create label and place the image
        f_lbl = Label(self.root, image=self.photoimg)
        f_lbl.place(x=0, y=0, width=530, height=190)

        #2nd image 
        img2 = Image.open("C:/Users/satya/Desktop/machine/project2final/image/face5.jpg")
        # Resize the image
        img2 = img2.resize((510, 190), Image.BILINEAR)  # Use BILINEAR as an example
         # Convert the resized image to PhotoImage
        self.photoimg2 = ImageTk.PhotoImage(img2)
        # Create label and place the image
        f_lbl2 = Label(self.root, image=self.photoimg)
        f_lbl2.place(x=540, y=0, width=530, height=190)


        #3rd image
        img3 = Image.open("C:/Users/satya/Desktop/machine/project2final/image/face6.jpg")
        # Resize the image
        img3 = img3.resize((510, 190), Image.BILINEAR)  # Use BILINEAR as an example
         # Convert the resized image to PhotoImage
        self.photoimg3 = ImageTk.PhotoImage(img3)
        # Create label and place the image
        f_lb3 = Label(self.root, image=self.photoimg)
        f_lb3.place(x=1080, y=0, width=510, height=190)
        
        #4th image
        img4 = Image.open("C:/Users/satya/Desktop/machine/project2final/image/plain.jpg")
        # Resize the image
        img4 = img4.resize((1530, 790), Image.BILINEAR)  # Use BILINEAR as an example
         # Convert the resized image to PhotoImage
        self.photoimg4 = ImageTk.PhotoImage(img4)
        # Create label and place the image
        bg_img = Label(self.root, image=self.photoimg)
        bg_img.place(x=0, y=200, width=1530, height=790)

        #title of the project
        title_lbl=Label(bg_img,text="FACE RECOGNITION ATTENDENCE SYSTEM",font=("times new romen",40,"bold"),bg="green",fg="red")
        title_lbl.place(x=0,y=0,width=1530,height=45)

        #create button
        img5 = Image.open("C:/Users/satya/Desktop/machine/project2final/image/face1.jpg")
        img5 = img5.resize((190, 190), Image.BILINEAR)  # Use BILINEAR as an example
        self.photoimg5 = ImageTk.PhotoImage(img5)

        b1=Button(bg_img,image=self.photoimg5,cursor="hand2")
        b1.place(x=200,y=100,width=220,height=220)

        b2=Button(bg_img,image=self.photoimg5,cursor="hand2")
        b2.place(x=600,y=100,width=220,height=220)
        b3=Button(bg_img,image=self.photoimg5,cursor="hand2")
        b3.place(x=1000,y=100,width=220,height=220)
        b4=Button(bg_img,image=self.photoimg5,cursor="hand2")
        b4.place(x=1400,y=100,width=220,height=220)

#++++++++++++=============================function+++++++++========
    def  student_details(self):
        self.new_window=Toplevel(self.root)
        self.app=Student(self.new_window)




























        

if __name__ == "__main__":
    root = Tk()
    obj = Face_recognition_system(root)
    root.mainloop()

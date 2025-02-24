from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import mysql.connector
import cv2
import os
import numpy as np
from time import strftime
from datetime import datetime

class Face_Recognition:
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
        b11 = Button(self.root, text="recognition face and attendence", command=self.face_recog, cursor="hand2", font=("times new roman", 20, "bold"), bg="blue", fg="white")

        #b11 = Button(self.root, text="recognition face bbu",command=self.face_recog ,cursor="hand2" ,font=("times new roman", 20, "bold"), bg="blue", fg="white")
        b11.place(x=1110, y=550,width=250, height=55)


        #Load and resize the second image
    def mark_Attendence(self,i,r,n,d,s):
        with open("Attendence.csv", "r+", newline="\n") as f:
            myDataList = f.readlines()
            name_list = []
            for line in myDataList:
                entry = line.strip().split((","))
                name_list.append(entry[0])
                # if name not in name_list:
                # f.writelines(f"\n{i},{r},{n},{d}")
                # Load and resize the second image
            if((str(i) not in name_list) and (n not in name_list)and(d not in name_list)and (r not in name_list)):
                now=datetime.now()
                # d1=now.strftime("%d/%m/%Y")
                d1 = now.strftime("%d/%m/%Y")
                dtString=now.strftime("%H:%M:%S")
                f.writelines(f"\n{i},{r},{n},{d},{dtString},{d1},{s},Present")





                













    def face_recog(self):
        def draw_boundry(img,classifier,scaleFactor,minNeighbbors,color,text,clf):
            gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            features=classifier.detectMultiScale(gray_image,scaleFactor,minNeighbbors)

            coord=[]
            for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                id,predict=clf.predict(gray_image[y:y+h,x:x+w])
                confidence=int((100*(1-predict/300)))



                conn=mysql.connector.connect(host="localhost",username="root",password="satyam",database="facerecognition")
                my_cursor=conn.cursor()

                # my_cursor.execute("select name from student where std_id="+str(id))
                # n=my_cursor.fetchone()
                # n="+".join(n)

                # my_cursor.execute("select std_id from student where std_id="+str(id))
                # i=my_cursor.fetchone()
                # i="+".join(i)

                # my_cursor.execute("select roll from student where std_id ="+str(id))
                # r=my_cursor.fetchone()
                # r="+".join(r)

                # my_cursor.execute("select dep from student where std_id="+str(id))
                # d=my_cursor.fetchone()
                # d="+".join(d)
                
                my_cursor.execute("SELECT name FROM student WHERE std_id=%s", (str(id),))
                n = my_cursor.fetchone()
                n = "+".join(map(str, n)) if n else "Unknown"

                my_cursor.execute("SELECT std_id FROM student WHERE std_id=%s", (str(id),))
                i = my_cursor.fetchone()
                i = "+".join(map(str, i)) if i else "Unknown"

                my_cursor.execute("SELECT roll_no FROM student WHERE std_id=%s", (str(id),))
                r = my_cursor.fetchone()
                r = "+".join(map(str, r)) if r else "Unknown"

                my_cursor.execute("SELECT dep FROM student WHERE std_id=%s", (str(id),))
                d = my_cursor.fetchone()
                d = "+".join(map(str, d)) if d else "Unknown"

                my_cursor.execute("SELECT sem FROM student WHERE std_id=%s", (str(id),))
                s = my_cursor.fetchone()
                s = "+".join(map(str, r)) if r else "Unknown"



                if confidence>77:
                    # cv2.putText(img,f"id:{i}",(x,y+55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),4)
                    cv2.putText(img,f"name:{n}",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),4)
                    cv2.putText(img,f"roll_no:{r}",(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),4)
                    cv2.putText(img,f"department:{d}",(x,y-105),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),4)
                    self.mark_Attendence(  i,r,n,d ,s)
                else:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(img,"Unknown Face",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),4)
                coord=[x,y,w,y]
            return coord
        faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  
        def recognize(img,clf,faceCascade):
            coord=draw_boundry(img,faceCascade,1.1,10,(255,25,255),"Face",clf)
            return img
          
       
        # clf=cv2.face.createLBPHFaceRecognizer()
        clf=cv2.face.LBPHFaceRecognizer_create()
        

        clf.read("classifier.xml")

        video_cap=cv2.VideoCapture(0)   
        while True:
            ret,img=video_cap.read()
            img=recognize(img,clf,faceCascade)
            cv2.imshow("Welcome to face Recognition",img)
            if cv2.waitKey(1)==13:  
                break
        video_cap.release()
        cv2.destroyAllWindows()
        


if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition(root)
    root.mainloop()

from tkinter import*
from tkinter import ttk
from tkinter import messagebox
from PIL import Image,ImageTk
import mysql.connector
import cv2
class Student:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recoginition System")




    def generate_dataset(self):
        if self.var_dep.get()=="Select Department" or self.var_std_name.get()=="" or self.var_std_id.get()=="":
            messagebox.showerror("Error","All fields are required",parent=self.root)

        else:
            try:
                conn=mysql.connector.connect(host="localhost",username="root",password="satyam",database="facerecognition")
                my_cursor=conn.cursor()
                sql="select * from student"
                # val=(self.var_std_id.get(),)
                my_cursor.execute(sql)
                myresult = my_cursor.fetchall()
                id=0
                for x in myresult:
                    id+=1
                
                # id = len(myresult)
                my_cursor.execute("update student set dep=%s,course=%s,year=%s,sem=%s,name=%s,division=%s,roll_no=%s,gender=%s,dob=%s,email=%s,phone=%s,address=%s,teacher=%s,photo=%s where std_id=%s",(
                                                                                                                self.var_dep.get(),
                                                                                                                self.var_course.get(),
                                                                                                                self.var_year.get(),
                                                                                                                self.var_semester.get(),
                                                                                                            
                                                                                                                self.var_std_name.get(),
                                                                                                                self.var_division.get(),
                                                                                                                self.var_roll.get(),
                                                                                                                self.var_gender.get(),
                                                                                                                self.var_dov.get(),
                                                                                                                self.var_email.get(),
                                                                                                                self.var_phone.get(),
                                                                                                                self.var_address.get(),
                                                                                                                self.var_teacher.get(),
                                                                                                                self.var_radio1.get() ,
                                                                                                                self.var_std_id.get()==id+1
                                                                                                                ))
                conn.commit()
                self.fetch_data()
                self.reset_data()
                conn.close()
                #======load predefined data on face frontal from open cv=====
                face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                def face_cropped(img):
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces=face_classifier.detectMultiScale(gray,1.3,5)
                    for (x,y,w,h) in faces:
                        face_cropped=img[y:y+h,x:x+w]
                        return face_cropped
                cap=cv2.VideoCapture(0)
                img_id=0
                while True:
                    ret,my_frame=cap.read()
                    if face_cropped(my_frame) is not None:
                        img_id+=1
                    face=cv2.resize(face_cropped(my_frame),(450,450))
                    face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                    file_name_path="data/user."+str(id)+"."+str(img_id)+".jpg"
                    cv2.imwrite(file_name_path)
                    cv2.putText(face,str(img_id),(50,50),cv2.Font_HERSHY_COMPLEX,2,(0,225,0),2)
                    cv2.imshow("Cropped Face",face)
                    if cv2.waitKey(1)==13 or img_id==100:
                        break
                cap.release()
                cv2.destroyAllWindows()
                messagebox.showinfo("result","generating data set completed")
            except Exception as es:
                messagebox.showerror("Error",f"Due to :{str(es)}",parent=self.root)
if __name__ == "__main__":
    root=Tk()
    root.title("Studen data")
    obj=Student(root)
    root.mainloop()                    

import cv2
import numpy as np
import sqlite3
import mysql.connector
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0); #open video camera
def insertorupdate(Id,Name,age):
   # conn=mysql.connector.connect(host="localhost",username="root",password="satyam",database="facerecognition")
   # my_cursor=conn.cursor()
   # my_cursor.execute("select * from student where id="+str(Id))
   # row=my_cursor.fetchone()
    # if row==None:
    #     my_cursor.execute("insert into student values("+str(Id)+",'"+Name+"','"+age+"')")
    # else:
    #     my_cursor.execute("update student set name='"+Name+"',age='"+age+"' where id="+str(Id))
    # conn.commit()
    # conn.close()
    conn=sqlite3.connect("sqlite.db")
   
    cmd="select * from students where id="+str(Id);
    
    cursor=conn.execute(cmd)

    isrecordExist=0;
    for row in cursor:
        isrecordExist=1;
    if isrecordExist==1:
        conn.execute("update students set Name=? where Id=?",(Name,Id))
        conn.execute("update students set age=? where Id=?",(age,Id))
     

    else:
        conn.execute("insert into students (Id,Name,age) values(?,?,?)",(Id,Name,age))

       
    conn.commit()
#inser user defined values in to table
Id=input('Enter User Id :')
Name=input('Enter User Name :')
age=input('Enter User Age :')
insertorupdate(Id,Name,age)
#take input from user detect face in webcamera coding

sampleNum=0
#take 100 images of user
while(True):
    ret, img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite("dataset/User."+str(Id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)#delayed time
    cv2.imshow("Face",img)
    cv2.waitKey(1)
    if(sampleNum>30):
        break
cam.release()
cv2.destroyAllWindows()
    
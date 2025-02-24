from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import mysql.connector
import cv2
import os
import numpy as np
import csv
mydata=[]

class Attendence:
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
        img_path1 = "image/face.jpg"
        img1 = Image.open(img_path1)
        img1 = img1.resize((510, 150), Image.BILINEAR)
        self.photoimg1 = ImageTk.PhotoImage(img1)
        f_lbl1 = Label(self.root, image=self.photoimg1)
        f_lbl1.place(x=0, y=0)

        # Load and resize the second image
        img_path2 = "image/face.jpg"
        img2 = Image.open(img_path2)
        img2 = img2.resize((510, 150), Image.BILINEAR)
        self.photoimg2 = ImageTk.PhotoImage(img2)
        f_lbl2 = Label(self.root, image=self.photoimg2)
        f_lbl2.place(x=540, y=0)

        # Load and resize the third image
        img_path3 = "image/face.jpg"
        img3 = Image.open(img_path3)
        img3 = img3.resize((510, 130), Image.BILINEAR)
        self.photoimg3 = ImageTk.PhotoImage(img3)
        f_lbl3 = Label(self.root, image=self.photoimg3)
        f_lbl3.place(x=1080, y=0)

        # Load and resize the background image
        bg_img_path = "C:/Users/satya/Desktop/machine/project2final/image/face6.jpg"
        bg_img = Image.open(bg_img_path)
        bg_img = bg_img.resize((1530, 790), Image.BILINEAR)
        self.photoimg4 = ImageTk.PhotoImage(bg_img)
        bg_lbl = Label(self.root, image=self.photoimg4)
        bg_lbl.place(x=0, y=130)

        title_lbl = Label(self.root, text="STUDENT ATTENDENCE SYSTEM", font=("times new roman", 26, "bold"), bg="green", fg="lightblue")
        title_lbl.place(x=0, y=130, width=1530, height=35)
        # frame
        main_frame = Frame(bg_lbl, bd=2, bg="light blue")
        main_frame.place(x=20,y=40,width=1480,height=790)

        #leftframe 
        left_frame = LabelFrame(main_frame, bd=2, bg="white", relief=RIDGE, text="Student Attendence Details",font=("times new roman", 16))
        left_frame.place(x=10, y=10, width=720, height=680)  # 'height' was misspelled as 'hight'
        # left_frame = LabelFrame(left_frame, bd=2, bg="white", relief=RIDGE, text="coursDetails",font=("times new roman", 16))
        # left_frame.place(x=10, y=10, width=700, height=180)  # 'height' was misspelled as 'hight'
     
        AttendenceId_label = Label(left_frame, text=" AttendenceId ", font=("times new roman", 12, "bold"), bg="white")
        AttendenceId_label.grid(row=0, column=0, padx=10, pady=10, sticky=W)

        Att_id_Entry=ttk.Entry(left_frame,width=22, font=("times new roman", 12, "bold"))
        Att_id_Entry.grid(row=0,column=1,pady=10,padx=20,sticky=W)

        
        # #roll
        roll_label = Label(left_frame, text="Roll no :", font=("times new roman", 12, "bold"), bg="white")
        roll_label.grid(row=0, column=2, padx=10, pady=10, sticky=W)

        roll_Entry=ttk.Entry(left_frame,width=22, font=("times new roman", 12, "bold"))
        roll_Entry.grid(row=0,column=3,pady=10,padx=20,sticky=W)

        # Name
        Name_label = Label(left_frame, text="Name :", font=("times new roman", 12, "bold"), bg="white")
        Name_label.grid(row=1, column=0, padx=10, pady=10, sticky=W)

        Name_Entry=ttk.Entry(left_frame,width=22, font=("times new roman", 12, "bold"))
        Name_Entry.grid(row=1,column=1,pady=10,padx=20,sticky=W)
        #Department
        Depart_label = Label(left_frame, text="Department :", font=("times new roman", 12, "bold"), bg="white")
        Depart_label.grid(row=1, column=2, padx=10, pady=10, sticky=W)

        
        DepartEntry=ttk.Entry(left_frame,width=22, font=("times new roman", 12, "bold"))
        DepartEntry.grid(row=1,column=3,pady=10,padx=20,sticky=W)
        
        #-----------------------------time
        time_label = Label(left_frame, text="time :", font=("times new roman", 12, "bold"), bg="white")
        time_label.grid(row=2, column=0, padx=10, pady=10, sticky=W)

        
        time_Entry=ttk.Entry(left_frame,width=22, font=("times new roman", 12, "bold"))
        time_Entry.grid(row=2,column=1,pady=10,padx=20,sticky=W)





        #-------------------------------------------- Date----
        Date_label = Label(left_frame, text="Date :", font=("times new roman", 12, "bold"), bg="white")
        Date_label.grid(row=2, column=2, padx=10, pady=10, sticky=W)

        
        Date_id_Entry=ttk.Entry(left_frame,width=22, font=("times new roman", 12, "bold"))
        Date_id_Entry.grid(row=2,column=3,pady=10,padx=20,sticky=W)



        #----------------------------------------------------
        semester_label = Label(left_frame, text="Semester", font=("times new roman", 12, "bold"), bg="white")
        semester_label.grid(row=3, column=0, padx=10, pady=10, sticky=W)

        
        student_id_Entry=ttk.Entry(left_frame,width=22, font=("times new roman", 12, "bold"))
        student_id_Entry.grid(row=3,column=1,pady=10,padx=20,sticky=W)
        
        #-----------------------------------
        Status_label = Label(left_frame, text="Attendence Status :", font=("times new roman", 12, "bold"), bg="white")
        Status_label.grid(row=4, column=0, padx=10, pady=10, sticky=W)

        Status_combo = ttk.Combobox(left_frame, font=("times new roman", 12, "bold"), state="readonly", width=20)
        Status_combo["values"] = ("Status","Absent", "Present")
        Status_combo.current(0)
        Status_combo.grid(row=4, column=1, padx=10, pady=10, sticky=W)
       
        btn_frame=Frame(left_frame,relief=RIDGE,bg="white")
        btn_frame.place(y=280,x=5,width=680,height=90)

        save_btn=Button(btn_frame,width=15,command=self.importCsv,font=("times new roman", 12, "bold"),text="Import CSV",bg="green",fg="white")
        save_btn.grid(row=0,column=0,padx=10,pady=6)
        update_btn=Button(btn_frame,width=15,font=("times new roman", 12, "bold"),text="Export CSV",bg="blue",fg="white")
        update_btn.grid(row=0,column=1,padx=10,pady=6)
        delete_btn=Button(btn_frame,width=15,font=("times new roman", 12, "bold"),text="Update",bg="yellow",fg="blue")
        delete_btn.grid(row=0,column=2,padx=10,pady=6)
        reset_btn=Button(btn_frame,width=15,font=("times new roman", 12, "bold"),text="Reset",bg="pink",fg="green")
        reset_btn.grid(row=0,column=3,padx=10,pady=6)
        



        # right frame
        reft_frame = LabelFrame(main_frame, bd=2, bg="white", relief=RIDGE,font=("times new roman", 16))
        reft_frame.place(x=740, y=10, width=720, height=680)
        table_frame=Frame(reft_frame,relief=RIDGE,bg="white")
        table_frame.place(y=5,x=5,width=680,height=490)
        scrol_x=ttk.Scrollbar(table_frame,orient=HORIZONTAL)
        scrol_y=ttk.Scrollbar(table_frame,orient=VERTICAL)
        self.AttendenceTable=ttk.Treeview(table_frame,column=("AttendenceId","Roll No ","Name","Department","time+","Date","Semester","Status"),xscrollcommand=scrol_x.set,yscrollcommand=scrol_y.set)
        scrol_x.pack(side=BOTTOM,fill=X)
        scrol_y.pack(side=RIGHT,fill=Y)
        scrol_x.config(command=self.AttendenceTable.xview)
        scrol_y.config(command=self.AttendenceTable.yview)
        self.AttendenceTable.heading("AttendenceId",text="AttendenceId")
        self.AttendenceTable.heading("Roll No ",text="Roll No ")
        self.AttendenceTable.heading("Name",text="Name")
        self.AttendenceTable.heading("Department",text="Department")
        self.AttendenceTable.heading("time+",text="time+")
        self.AttendenceTable.heading("Date",text="Date")
        self.AttendenceTable.heading("Semester",text="Semester")
        self.AttendenceTable.heading("Status",text="Status")
        self.AttendenceTable["show"]="headings"
        self.AttendenceTable.column("AttendenceId",width=100)
        self.AttendenceTable.column("Roll No ",width=100)
        self.AttendenceTable.column("Name",width=100)
        self.AttendenceTable.column("Department",width=100)
        self.AttendenceTable.column("time+",width=100)
        self.AttendenceTable.column("Date",width=100)
        self.AttendenceTable.column("Semester",width=100)
        self.AttendenceTable.column("Status",width=100)
        self.AttendenceTable.pack(fill=BOTH,expand=1)
        # self.AttendenceTable.bind("<ButtonRelease>",self.get_cursor)
        # self.fetch_data()

        #====================fetchdata============
    def fetchData(self,rows):
        self.AttendenceTable.delete(*self.AttendenceTable.get_children())
        for i in rows:
            self.AttendenceTable.insert("",END,values=i)
            # def get_cursor(self,event=""):
            #     cursor_row=self.AttendenceTable.focus()
            #     content=self.AttendenceTable.item(cursor_row)
            #     row=content["values"]
            #     self.txtAttendenceId.delete(0,END)
            #     self.txtAttendenceId.insert(END,row[0])

    def importCsv(self):
        global mydata
        filename=filedialog.askopenfilename(initialdir=os.getcwd(),title="Open CSV",filetypes=(("CSV file","*.csv"),("All Files","*.*")),parent=self.root)
        with open(filename) as myfile:
            csvread=csv.reader(myfile,delimiter=",")
            for i in csvread:
                mydata.append(i)
            self.fetchData(mydata)
    




        



if __name__ == "__main__":
    root = Tk()
    obj = Attendence(root)
    root.mainloop()

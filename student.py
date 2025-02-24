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
#++++++Variable++++++++
        self.var_dep=StringVar()
        self.var_phone=StringVar()
        self.var_course = StringVar()
        self.var_year = StringVar()
        self.var_semester = StringVar()
        self.var_std_id = StringVar()
        self.var_std_name = StringVar()
        self.var_roll = StringVar()
        self.var_gender = StringVar()
        self.var_email = StringVar()
        self.var_address = StringVar()
        self.var_teacher = StringVar()
        self.var_dov=StringVar()
        self.var_division=StringVar()










        img_path1 = "C:/Users/satya/Desktop/Examprep/machine/project2final/image/face.jpg"
        img1 = Image.open(img_path1)
        img1 = img1.resize((510, 150), Image.BILINEAR)
        self.photoimg1 = ImageTk.PhotoImage(img1)
        f_lbl1 = Label(self.root, image=self.photoimg1)
        f_lbl1.place(x=0, y=0)

        # Load and resize the second image
        img_path2 = "C:/Users/satya/Desktop/Examprep/machine/project2final/image/face.jpg"
        img2 = Image.open(img_path2)
        img2 = img2.resize((510, 150), Image.BILINEAR)
        self.photoimg2 = ImageTk.PhotoImage(img2)
        f_lbl2 = Label(self.root, image=self.photoimg2)
        f_lbl2.place(x=540, y=0)

        # Load and resize the third image
        img_path3 = "C:/Users/satya/Desktop/Examprep/machine/project2final/image/face.jpg"
        img3 = Image.open(img_path3)
        img3 = img3.resize((510, 130), Image.BILINEAR)
        self.photoimg3 = ImageTk.PhotoImage(img3)
        f_lbl3 = Label(self.root, image=self.photoimg3)
        f_lbl3.place(x=1080, y=0)

        # Load and resize the background image
        bg_img_path = "C:/Users/satya/Desktop/Examprep/machine/project2final/image/face6.jpg"
        bg_img = Image.open(bg_img_path)
        bg_img = bg_img.resize((1530, 790), Image.BILINEAR)
        self.photoimg4 = ImageTk.PhotoImage(bg_img)
        bg_lbl = Label(self.root, image=self.photoimg4)
        bg_lbl.place(x=0, y=130)

        title_lbl = Label(self.root, text="STUDENT MANAGEMENT SYSTEM", font=("times new roman", 26, "bold"), bg="green", fg="lightblue")
        title_lbl.place(x=0, y=130, width=1530, height=35)
        # frame
        main_frame = Frame(bg_lbl, bd=2, bg="light blue")
        main_frame.place(x=20,y=40,width=1480,height=790)

        #leftframe 
        left_frame = LabelFrame(main_frame, bd=2, bg="white", relief=RIDGE, text="Student Details",font=("times new roman", 16))
        left_frame.place(x=10, y=10, width=720, height=680)  # 'height' was misspelled as 'hight'
        # left_frame = LabelFrame(left_frame, bd=2, bg="white", relief=RIDGE, text="coursDetails",font=("times new roman", 16))
        # left_frame.place(x=10, y=10, width=700, height=180)  # 'height' was misspelled as 'hight'
        current_course_frame=LabelFrame(left_frame, bd=2,bg="white",relief=RIDGE, text="Current Course information",font=("times new roman", 16))  
        current_course_frame.place(x=5,y=20,width=700,height=140)

        # dep_lable=Label(left_frame,text="department",font=("times new roman", 12,"bold"),bg="white")
        # dep_lable.grid(row=0,column=0 ,padx=10,pady=10,sticky=W)

        # dep_combo=ttk.Combobox(left_frame, font=("times new roman", 12,"bold") ,state="randomly")
        
        # dep_combo["values"]=("select Department","CSE","IT","CIVIL","MECHNICAL","ECE")
        # dep_combo.current(0)
        # dep_combo.grid(row=0,column=0,padx=4,pady=10,sticky=W)
        
        dep_label = Label(current_course_frame, text="Department", font=("times new roman", 12, "bold"), bg="white")
        dep_label.grid(row=0, column=0, padx=10, pady=10, sticky=W)

        dep_combo = ttk.Combobox(current_course_frame,textvariable=self.var_dep, font=("times new roman", 12, "bold"), state="readonly", width=20)
        dep_combo["values"] = ("Select Department", "CSE", "IT", "CIVIL", "MECHANICAL", "ECE")
        dep_combo.current(0)
        dep_combo.grid(row=0, column=1, padx=10, pady=10, sticky=W)
        # #course
        course_label = Label(current_course_frame, text="Cource", font=("times new roman", 12, "bold"), bg="white")
        course_label.grid(row=0, column=2, padx=10, pady=10, sticky=W)

        course_combo = ttk.Combobox(current_course_frame,textvariable=self.var_course, font=("times new roman", 12, "bold"), state="readonly", width=20)
        course_combo["values"] = ("Select Course", "FE", "SE", "TE", "BE")
        course_combo.current(0)
        course_combo.grid(row=0, column=3, padx=10, pady=10, sticky=W)
       
        # year frame
        year_label = Label(current_course_frame, text="  Year", font=("times new roman", 12, "bold"), bg="white")
        year_label.grid(row=1, column=0, padx=10, pady=10, sticky=W)

        year_combo = ttk.Combobox(current_course_frame,textvariable=self.var_year ,font=("times new roman", 12, "bold"), state="readonly", width=20)
        year_combo["values"] = ("Select Year", "2022", "2023", "2024", "2025", "2026", "2027", "2028")
        year_combo.current(0)
        year_combo.grid(row=1, column=1, padx=10, pady=10, sticky=W)
        #semester
        semester_label = Label(current_course_frame, text="Semester", font=("times new roman", 12, "bold"), bg="white")
        semester_label.grid(row=1, column=2, padx=10, pady=10, sticky=W)

        semester_combo = ttk.Combobox(current_course_frame,textvariable=self.var_semester ,font=("times new roman", 12, "bold"), state="readonly", width=20)
        semester_options = ("Select Semester", "1st Semester", "2nd Semester", "3rd Semester", "4th Semester", "5th Semester", "6th Semester", "7th Semester", "8th Semester")

        semester_combo["values"] =semester_options 
        semester_combo.current(0)
        semester_combo.grid(row=1, column=3, padx=10, pady=10, sticky=W)
        #class student information
        class_student_frame=LabelFrame(left_frame, bd=2,bg="white",relief=RIDGE, text="Class Student information",font=("times new roman", 16))  
        class_student_frame.place(x=5,y=160,width=700,height=400)
        #student id
        studentId_label=Label(class_student_frame, text="Student_Id", font=("times new roman", 12, "bold"), bg="white")
        studentId_label.grid(row=0,column=0,pady=10 ,padx=10,sticky=W)
        student_id_Entry=ttk.Entry(class_student_frame,textvariable=self.var_std_id,width=22, font=("times new roman", 12, "bold"))
        student_id_Entry.grid(row=0,column=1,pady=10,padx=20,sticky=W)
        #student name
        studentId_label=Label(class_student_frame, text="Student Name", font=("times new roman", 12, "bold"), bg="white")
        studentId_label.grid(row=0,column=2,pady=10 ,padx=10,sticky=W)
        student_id_Entry=ttk.Entry(class_student_frame,textvariable=self.var_std_name,width=22, font=("times new roman", 12, "bold"))
        student_id_Entry.grid(row=0,column=3,pady=10,padx=20,sticky=W)
        #ClassDivision
        ClassDivision_label=Label(class_student_frame, text="ClassDivision", font=("times new roman", 12, "bold"), bg="white")
        ClassDivision_label.grid(row=1,column=0 ,padx=10,pady=10,sticky=W)
        # ClassDivision_labelEntry=ttk.Entry(class_student_frame,textvariable=self.var_division,width=22, font=("times new roman", 12, "bold"))
        # ClassDivision_labelEntry.grid(row=1,column=1,pady=10,padx=20,sticky=W)
        ClassDivision_combo = ttk.Combobox(class_student_frame,textvariable=self.var_division ,font=("times new roman", 12, "bold"), state="readonly", width=20)
        ClassDivision_options = ("I", "II", "III","IV")

        ClassDivision_combo["values"] =ClassDivision_options 
        ClassDivision_combo.current(0)
        ClassDivision_combo.grid(row=1, column=1, padx=20, pady=10, sticky=W)
       
        #RollNo
        RollNo_label=Label(class_student_frame, text="RollNo:", font=("times new roman", 12, "bold"), bg="white")
        RollNo_label.grid(row=1,column=2,padx=10,pady=10,sticky=W)
        RollNo_Entry=ttk.Entry(class_student_frame,textvariable=self.var_roll,width=22, font=("times new roman", 12, "bold"))
        RollNo_Entry.grid(row=1,column=3,padx=20,pady=10,sticky=W)
        #gender
        gender_label=Label(class_student_frame, text="Gender:", font=("times new roman", 12, "bold"), bg="white")
        gender_label.grid(row=2,column=0 ,pady=10,padx=10,sticky=W)
        gender_combo = ttk.Combobox(class_student_frame,textvariable=self.var_gender ,font=("times new roman", 12, "bold"), state="readonly", width=20)
        gender_options = ("Male", "Female", "other")

        gender_combo["values"] =gender_options 
        gender_combo.current(0)
        gender_combo.grid(row=2, column=1, padx=20, pady=10, sticky=W)
       
        
        # #Dob
        Dob_label=Label(class_student_frame, text="DOB:", font=("times new roman", 12, "bold"), bg="white")
        Dob_label.grid(row=2,column=2,padx=10,pady=10,sticky=W)
        Dob_id_Entry=ttk.Entry(class_student_frame,textvariable=self.var_dov,width=22, font=("times new roman", 12, "bold"))
        Dob_id_Entry.grid(row=2,column=3,padx=20,pady=10,sticky=W)
        #Email
        Email_label=Label(class_student_frame, text="Email", font=("times new roman", 12, "bold"), bg="white")
        Email_label.grid(row=3,column=0 ,padx=10,pady=10,sticky=W)
        Email_id_Entry=ttk.Entry(class_student_frame,textvariable=self.var_email,width=22, font=("times new roman", 12, "bold"))
        Email_id_Entry.grid(row=3,column=1,padx=20,pady=10,sticky=W)
        #PhoneNo
        PhoneNo_label=Label(class_student_frame, text="Phone No:", font=("times new roman", 12, "bold"), bg="white")
        PhoneNo_label.grid(row=3,column=2,padx=10,pady=10,sticky=W)
        PhoneNo_Entry=ttk.Entry(class_student_frame,textvariable=self.var_phone,width=22, font=("times new roman", 12, "bold"))
        PhoneNo_Entry.grid(row=3,column=3,padx=20,pady=10,sticky=W)
        #Address
        Address_label=Label(class_student_frame, text="Address:", font=("times new roman", 12, "bold"), bg="white")
        Address_label.grid(row=4,column=0 ,padx=10,pady=10,sticky=W)
        Address_Entry=ttk.Entry(class_student_frame,textvariable=self.var_address,width=22, font=("times new roman", 12, "bold"))
        Address_Entry.grid(row=4,column=1,padx=20,pady=10,sticky=W)
        #teacher
        Techar_Name_label=Label(class_student_frame, text="Techar_Name:", font=("times new roman", 12, "bold"), bg="white")
        Techar_Name_label.grid(row=4,column=2,padx=10,pady=10,sticky=W)
        Techar_Name_Entry=ttk.Entry(class_student_frame,textvariable=self.var_teacher,width=22, font=("times new roman", 12, "bold"))
        Techar_Name_Entry.grid(row=4,column=3,padx=20,pady=10,sticky=W)
        #radio button
        self.var_radio1=StringVar()
        
        radiobtn1=ttk.Radiobutton(class_student_frame,text="Take photo",value="Yes",variable=self.var_radio1)
        radiobtn1.grid(row=5,column=0,pady=10)

        radiobtn2=ttk.Radiobutton(class_student_frame,text="Take Without photo",variable=self.var_radio1,value="No")
        radiobtn2.grid(row=5,column=1,pady=10)

        #buttons frame
        btn_frame=Frame(class_student_frame,bd=2,relief=RIDGE,bg="white")
        btn_frame.place(y=280,x=5,width=680,height=90)

        save_btn=Button(btn_frame,command=self.add_data,width=15,font=("times new roman", 12, "bold"),text="Save",bg="green",fg="white")
        save_btn.grid(row=0,column=0,padx=10,pady=6)
        update_btn=Button(btn_frame,command=self.update_data,width=15,font=("times new roman", 12, "bold"),text="Update",bg="blue",fg="white")
        update_btn.grid(row=0,column=1,padx=10,pady=6)
        delete_btn=Button(btn_frame,width=15,command=self.delete_data,font=("times new roman", 12, "bold"),text="delete",bg="yellow",fg="blue")
        delete_btn.grid(row=0,column=2,padx=10,pady=6)
        reset_btn=Button(btn_frame,width=15,command=self.reset_data,font=("times new roman", 12, "bold"),text="Reset",bg="pink",fg="green")
        reset_btn.grid(row=0,column=3,padx=10,pady=6)
        take_photo_btn=Button(btn_frame,command=self.generate_dataset,width=15,font=("times new roman", 12, "bold"),text="take photo",bg="yellow",fg="blue")
        take_photo_btn.grid(row=1,column=2,padx=10,pady=6)
        resetPhoto_btn=Button(btn_frame,width=15,font=("times new roman", 12, "bold"),text="Reset photo",bg="pink",fg="green")
        resetPhoto_btn.grid(row=1,column=1,padx=10,pady=6)

        # #right frame
        reft_frame = LabelFrame(main_frame, bd=2, bg="white", relief=RIDGE,font=("times new roman", 16))
        reft_frame.place(x=740, y=10, width=720, height=680)  # 'height' was misspelled as 'hight'
        # current cource 
        #search student
        search_frame=LabelFrame(reft_frame, bd=2,bg="white",relief=RIDGE, text="Search System",font=("times new roman", 16))  
        search_frame.place(x=5,y=10,width=700,height=80)
        searchlevel =Label(search_frame, text="Search By:", font=("times new roman", 12, "bold"), bg="white")
        #searchlevel.place(x=740, y=10, width=720, height=680)  # 'height' was misspelled as 'hight'
        searchlevel.grid(row=0,column=0,padx=10,pady=10,sticky=W)
        search_combo = ttk.Combobox(search_frame, font=("times new roman", 12, "bold"), state="readonly", width=10)
        search_combo["values"] = ("Select ", "Roll No", "Phone no", "Email", "Name", "2026", "2027", "2028")
        search_combo.current(0)
        search_combo.grid(row=0, column=1, padx=5, pady=10, sticky=W)
        #entry
        Search_Entry=ttk.Entry(search_frame,width=22, font=("times new roman", 12, "bold"))
        Search_Entry.grid(row=0,column=2,padx=10,pady=10,sticky=W)
      #Button
        search_btn=Button(search_frame,width=15,font=("times new roman", 12, "bold"),text="search",bg="blue",fg="white")
        search_btn.grid(row=0,column=3,padx=5,pady=6)
        show_btn=Button(search_frame,width=10,font=("times new roman", 12, "bold"),text=" show All",bg="yellow",fg="blue")
        show_btn.grid(row=0,column=4,padx=5,pady=6)
      # self.student_table.pack(fill=BOTH, expand=1)
        # table frame
        table_frame = Frame(reft_frame, bd=2, bg="white", relief=RIDGE)
        table_frame.place(x=5, y=90, width=700, height=400)

        # Create horizontal and vertical scrollbars
        scrollx = ttk.Scrollbar(table_frame, orient=HORIZONTAL)
        scrolly = ttk.Scrollbar(table_frame, orient=VERTICAL)

        # Define columns for the Treeview
        columns = ("dep", "course", "year", "sem", "id", "name", "div","roll no","gender", "dob", "email","phone" ,"address", "teacher", "photo")

        # Create the ttk Treeview widget
        self.student_table = ttk.Treeview(table_frame, columns=columns, xscrollcommand=scrollx.set, yscrollcommand=scrolly.set)

        # Configure scrollbars to control the Treeview
        scrollx.config(command=self.student_table.xview)
        scrolly.config(command=self.student_table.yview)

        # Add headings to the Treeview
        self.student_table.heading("dep", text="Department")
        self.student_table.heading("course", text="Course")
        self.student_table.heading("year", text="Year")
        self.student_table.heading("sem", text="Semester")
        self.student_table.heading("id", text="StudentId")
        self.student_table.heading("name", text="Name")
        self.student_table.heading("div", text="Division")
        self.student_table.heading("roll no", text="Roll No")
        self.student_table.heading("gender", text="Gender")
        self.student_table.heading("dob", text="DOB")
        self.student_table.heading("email", text="Email"),
        self.student_table.heading("phone", text="Phone")
        self.student_table.heading("address", text="Address")
        self.student_table.heading("teacher", text="Teacher")
        self.student_table.heading("photo", text="PhotoSampleStatus")

        # Set Treeview to display headings
        self.student_table["show"] = "headings"
        self.student_table.column("dep", width=100)
        self.student_table.column("course", width=100)
        self.student_table.column("year", width=100)
        self.student_table.column("sem", width=100)
        self.student_table.column("id", width=100)  # Corrected column identifier
        self.student_table.column("name", width=100)
        self.student_table.column("div", width=100)
        self.student_table.column("roll no", width=100)
        self.student_table.column("gender", width=100)
        self.student_table.column("dob", width=100)
        self.student_table.column("email", width=100)
        self.student_table.column("phone", width=100)
        self.student_table.column("address", width=100)
        self.student_table.column("teacher", width=100)
        self.student_table.column("photo", width=150) 

        # Pack the Treeview widget and scrollbars into the table_frame
        self.student_table.pack(fill=BOTH, expand=1)

        self.student_table.bind("<ButtonRelease>",self.get_cursor)
        self.fetch_data()
        scrollx.pack(side=BOTTOM, fill=X)
        scrolly.pack(side=RIGHT, fill=Y)
    # Bind the treeview widget to the scrollbars

    def add_data(self):
        if self.var_dep.get()=="Select Department" or self.var_std_name.get()=="" or self.var_std_id=="":
            messagebox.showerror("Warning....🥰", "All fields are required", parent=self.root)
        else:
            try:
                conn=mysql.connector.connect(host="localhost",username="root",password="satyam",database="facerecognition")
                my_cursor=conn.cursor()
                my_cursor.execute("insert into student values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(
                                                                                                                self.var_dep.get(),
                                                                                                                self.var_course.get(),
                                                                                                                self.var_year.get(),
                                                                                                                self.var_semester.get(),
                                                                                                                self.var_std_id.get(),
                                                                                                                self.var_std_name.get(),
                                                                                                                self.var_division.get(),
                                                                                                                self.var_roll.get(),
                                                                                                                self.var_gender.get(),
                                                                                                                self.var_dov.get(),
                                                                                                                self.var_email.get(),
                                                                                                                self.var_phone.get(),
                                                                                                                self.var_address.get(),
                                                                                                                self.var_teacher.get(),
                                                                                                                self.var_radio1.get() ))
                 
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("Success","Student details has been added successfully",parent=self.root)
            except Exception as es:
                messagebox.showerror("Error",f"Due to:{str(es)}",parent=self.root)


#++++++++++++++++++++++++fetch   Data+++++++++++++++
    def fetch_data(self):
        conn=mysql.connector.connect(host="localhost",username="root",password="satyam",database="facerecognition")
        my_cursor=conn.cursor()
        my_cursor.execute("select * from student")
        data=my_cursor.fetchall()
        if len(data)!=0:
            self.student_table.delete(*self.student_table.get_children())
            for i in data:
                self.student_table.insert("",END,values=i)
            conn.commit()
        conn.close()

   #===================function get cursor=====================
    def get_cursor(self,event=""):
        cursor_focus=self.student_table.focus()
        content=self.student_table.item(cursor_focus)
        data=content["values"]
        self.var_dep.set(data[0]),  
        self.var_course.set(data[1]),
        self.var_year.set(data[2]),
        self.var_semester.set(data[3]),
        self.var_std_id.set(data[4]),
        self.var_std_name.set(data[5]),
        self.var_division.set(data[6]),
        self.var_roll.set(data[7]),
        self.var_gender.set(data[8]),
        self.var_dov.set(data[9]),
        self.var_email.set(data[10]),
        self.var_phone.set(data[11]),
        self.var_address.set(data[12]),
        self.var_teacher.set(data[13]),
        self.var_radio1.set(data[14])
#++++++++================update function ==========
    def update_data(self):
        if self.var_dep.get()=="Select Department" or self.var_std_name.get()=="" or self.var_std_id.get()=="":
            messagebox.showerror("Error","All fields are required",parent=self.root)
        else:
            try:
                update=messagebox.askyesno("update","Do you want to upadate this student details",parent=self.root)
                if update>0:
                    conn=mysql.connector.connect(host="localhost",username="root",password="satyam",database="facerecognition")
                    my_cursor=conn.cursor()
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
                                                                                                                 self.var_std_id.get()
                
                                                                                                                ))
                else:
                    if not update:
                        return
                messagebox.showinfo("success","student details successfully updated",parent=self.root)
                conn.commit()
                self.fetch_data()
                conn.close()
            except Exception as es:
                messagebox.showerror("Error",f"Due to :{str(es)}",parent=self.root)
                 
    #+++=============delete function=============
    def delete_data(self):
        if self.var_std_id.get()=="":
            messagebox.showerror("Error","Student id must be required",parent=self.root)
        else:
            try:
                delete=messagebox.askyesno("Student Delete Page","Do you want to delete the student",parent=self.root)
                if delete>0:
                    conn=mysql.connector.connect(host="localhost",username="root",password="satyam",database="facerecognition")
                    my_cursor=conn.cursor()
                    sql="delete from student where std_id=%s"
                    val=(self.var_std_id.get(),)
                    my_cursor.execute(sql,val)
                else:
                    if not delete:
                        return
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("Delete","Successfully deleted student details",parent=self.root)
            except Exception as es:
                messagebox.showerror("Error",f"Due to :{str(es)}",parent=self.root)
     #=============reset data======================
    def reset_data(self):
        self.var_dep.set("Select Department"),  
        self.var_course.set("Select Course"),
        self.var_year.set("Select Year"),
        self.var_semester.set("Select Semeter"),
        self.var_std_id.set(""),
        self.var_std_name.set(""),
        self.var_division.set("Select Division"),
        self.var_roll.set(""),
        self.var_gender.set("Male"),
        self.var_dov.set(""),
        self.var_email.set(""),
        self.var_phone.set(""),
        self.var_address.set(""),
        self.var_teacher.set(""),
        self.var_radio1.set("")


    #++++++++++++++++========================== Generate dataset and Take a photo Sample=====
    # def generate_dataset(self):
    #     if self.var_dep.get()=="Select Department" or self.var_std_name.get()=="" or self.var_std_id.get()=="":
    #         messagebox.showerror("Error","All fields are required",parent=self.root)

    #     else:
    #         try:
    #             conn=mysql.connector.connect(host="localhost",username="root",password="satyam",database="facerecognition")
    #             my_cursor=conn.cursor()
    #             sql="select * from student"
    #             # val=(self.var_std_id.get(),)
    #             my_cursor.execute(sql)
    #             # id=0
    #             # for x in myresult:
    #             #     id+=1
    #             myresult = my_cursor.fetchall()
    #             id = len(myresult)
    #             my_cursor.execute("update student set dep=%s,course=%s,year=%s,sem=%s,name=%s,division=%s,roll_no=%s,gender=%s,dob=%s,email=%s,phone=%s,address=%s,teacher=%s,photo=%s where std_id=%s",(
    #                                                                                                             self.var_dep.get(),
    #                                                                                                             self.var_course.get(),
    #                                                                                                             self.var_year.get(),
    #                                                                                                             self.var_semester.get(),
                                                                                                               
    #                                                                                                             self.var_std_name.get(),
    #                                                                                                             self.var_division.get(),
    #                                                                                                             self.var_roll.get(),
    #                                                                                                             self.var_gender.get(),
    #                                                                                                             self.var_dov.get(),
    #                                                                                                             self.var_email.get(),
    #                                                                                                             self.var_phone.get(),
    #                                                                                                             self.var_address.get(),
    #                                                                                                             self.var_teacher.get(),
    #                                                                                                             self.var_radio1.get() ,
    #                                                                                                             self.var_std_id.get()
    #                                                                                                             ))
    #             conn.commit()
    #             self.fetch_data()
    #             self.reset_data()
    #             conn.close()
    #             #======load predefined data on face frontal from open cv=====
    #             face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    #             def face_cropped(img):
    #                 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #                 faces=face_classifier.detectMultiScale(gray,1.3,5)
    #                 for (x,y,w,h) in faces:
    #                     face_cropped=img[y:y+h,x:x+w]
    #                     return face_cropped
    #             cap=cv2.VideoCapture(0)
    #             img_id=0
    #             while True:
    #                 ret,my_frame=cap.read()
    #                 if face_cropped(my_frame) is not None:
    #                     img_id+=1
    #                     face=cv2.resize(face_cropped(my_frame),(450,450))
    #                     face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    #                     file_name_path="data/user."+str(id)+"."+str(img_id)+".jpg"
    #                     cv2.imwrite(file_name_path)
    #                     cv2.putText(face,str(img_id),(50,50),cv2.Font_HERSHY_COMPLEX,2,(0,225,0),2)
    #                     cv2.imshow("Cropped Face",face)
    #                 if cv2.waitKey(1)==13 or img_id==100:
    #                     break
    #             cap.release()
    #             cv2.destroyAllWindows()
    #             messagebox.showinfo("result","generating data set completed")
    #         except Exception as es:
    #             messagebox.showerror("Error",f"Due to :{str(es)}",parent=self.root)
    def generate_dataset(self):
        if self.var_dep.get() == "Select Department" or self.var_std_name.get() == "" or self.var_std_id.get() == "":
            messagebox.showerror("Error", "All fields are required", parent=self.root)
        else:
            try:
                conn = mysql.connector.connect(host="localhost", username="root", password="satyam", database="facerecognition")
                my_cursor = conn.cursor()

                # Fetch existing student records
                sql = "SELECT * FROM student"
                my_cursor.execute(sql)
                myresult = my_cursor.fetchall()
                id = len(myresult)  # Get the number of existing records

                # Update the student record
                update_query = "UPDATE student SET dep=%s, course=%s, year=%s, sem=%s, name=%s, division=%s, roll_no=%s, gender=%s, dob=%s, email=%s, phone=%s, address=%s, teacher=%s, photo=%s WHERE std_id=%s"
                my_cursor.execute(update_query, (
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
                    self.var_radio1.get(),
                    self.var_std_id.get()
                ))

                conn.commit()  # Commit the transaction

                # Fetch and reset data
                self.fetch_data()
                self.reset_data()

                conn.close()

                # Load predefined data on face frontal from OpenCV
                face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

                def face_cropped(img):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        face_cropped = img[y:y + h, x:x + w]
                        return face_cropped

                cap = cv2.VideoCapture(0)
                img_id = 0
                while True:
                    ret, my_frame = cap.read()
                    if face_cropped(my_frame) is not None:
                        img_id += 1
                        face = cv2.resize(face_cropped(my_frame), (450, 450))
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        file_name_path = f"data/user.{id}.{img_id}.jpg"
                        cv2.imwrite(file_name_path, face)
                        cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 225, 0), 2)
                        cv2.imshow("Cropped Face", face)
                    if cv2.waitKey(1) == 13 or img_id == 100:
                        break

                cap.release()
                cv2.destroyAllWindows()

                messagebox.showinfo("Result", "Generating dataset completed")

            except Exception as es:
                messagebox.showerror("Error", f"Due to: {str(es)}", parent=self.root)
    

#=================================== Function button===============
   










                   

        







        




if __name__ == "__main__":
    root=Tk()
    root.title("Studen data")
    obj=Student(root)
    root.mainloop()
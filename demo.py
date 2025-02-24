import cv2
import mysql.connector

class FaceRecognition:
    def face_recog(self):
        def draw_boundry(img, classifier, scaleFactor, minNeighbors, color, text, clf, conn):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
            coord = []
            for (x, y, w, h) in features:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                id, predict = clf.predict(gray_image[y:y + h, x:x + w])
                confidence = int((100 * (1 - predict / 300)))

                my_cursor = conn.cursor()

                my_cursor.execute("SELECT name FROM student WHERE std_id=%s", (str(id),))
                n = my_cursor.fetchone()
                n = "+".join(map(str, n)) if n else "Unknown"

                my_cursor.execute("SELECT std_id FROM student WHERE std_id=%s", (str(id),))
                i = my_cursor.fetchone()
                i = "+".join(map(str, i)) if i else "Unknown"

                my_cursor.execute("SELECT roll FROM student WHERE std_id=%s", (str(id),))
                r = my_cursor.fetchone()
                r = "+".join(map(str, r)) if r else "Unknown"

                my_cursor.execute("SELECT dep FROM student WHERE std_id=%s", (str(id),))
                d = my_cursor.fetchone()
                d = "+".join(map(str, d)) if d else "Unknown"

                if confidence > 77:
                    cv2.putText(img, f"id: {i}", (x, y - 70), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(img, f"name: {n}", (x, y - 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(img, f"roll: {r}", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(img, f"department: {d}", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                    self.mark_Attendence(i, r, n, d)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(img, "Unknown Face", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                coord = [x, y, w, h]
            return coord

        def recognize(img, clf, faceCascade, conn):
            coord = draw_boundry(img, faceCascade, 1.1, 10, (255, 25, 255), "Face", clf, conn)
            return img

        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

        conn = mysql.connector.connect(host="localhost", username="root", password="satyam", database="facerecognition")

        video_cap = cv2.VideoCapture(0)
        while True:
            ret, img = video_cap.read()
            if not ret:
                break
            img = recognize(img, clf, faceCascade, conn)
            cv2.imshow("Welcome to face Recognition", img)
            if cv2.waitKey(1) == 13:  # Enter key to break
                break
        video_cap.release()
        conn.close()
        cv2.destroyAllWindows()

    def mark_Attendence(self, i, r, n, d):
        # Implement the attendance marking logic here
        pass

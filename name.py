import cv2
import psycopg2
import os

def get_unique_id(person_name):
    try:
        connection = psycopg2.connect(
            dbname="your_db_name",
            user="your_username",
            password="your_password",
            host="your_server_host",
            port="your_port"
        )
        cursor = connection.cursor()
        query = "SELECT unique_id FROM persons WHERE name = %s;"
        cursor.execute(query, (person_name,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            print("No record found for the given name.")
            return None
    except Exception as e:
        print("Error while connecting to database:", e)
        return None
    finally:
        if connection:
            cursor.close()
            connection.close()

def capture_images(person_id, person_name):
    folder_path = f"dataset/{person_id}_{person_name}"
    os.makedirs(folder_path, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        img_path = os.path.join(folder_path, f"{person_id}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        cv2.imshow("Capturing Images", frame)
        count += 1
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {person_name} with ID {person_id}")

if __name__ == "__main__":
    person_name = input("Enter the name of the student/teacher: ")
    person_id = get_unique_id(person_name)
    
    if person_id:
        capture_images(person_id, person_name)
    else:
        print("Could not retrieve unique ID. Exiting...")

import os
import cv2
import sqlite3
import numpy as np
from datetime import datetime

# -------------------
# DATABASE SETUP
# -------------------
DATABASE = 'face_database.db'

def init_db():
    """Initialize the database and create tables if they don't exist."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                face_image_path TEXT NOT NULL
            )
        ''')
        conn.commit()

init_db()

def insert_user(user_name, user_id, face_image_path):
    """Insert a new user with name, ID, and face image path into the database."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (user_name, user_id, face_image_path) VALUES (?, ?, ?)',
                       (user_name, user_id, face_image_path))
        conn.commit()

def get_all_users():
    """Fetch all user data (name, id, face_image_path)."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT user_name, user_id, face_image_path FROM users')
        rows = cursor.fetchall()
    return rows

# -------------------
# GLOBAL VARIABLES
# -------------------
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# -------------------
# TRAIN RECOGNIZER
# -------------------
def train_recognizer():
    """Train the LBPH face recognizer with stored user data."""
    users = get_all_users()
    face_samples = []
    user_ids = []

    for user in users:
        user_name, user_id, face_image_path = user
        face_image = cv2.imread(face_image_path, cv2.IMREAD_GRAYSCALE)
        if face_image is not None:
            face_samples.append(face_image)
            user_ids.append(int(user_id))  # Use numeric IDs for the recognizer

    if face_samples:
        recognizer.train(face_samples, np.array(user_ids))
        print("Recognizer trained successfully!")

train_recognizer()

# -------------------
# REAL-TIME RECOGNITION
# -------------------
def recognize_and_display():
    """Detect faces, recognize them if known, and display their details."""
    users = {int(row[1]): row[0] for row in get_all_users()}  # Map user_id to user_name
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            user_id, confidence = recognizer.predict(face)
            
            if confidence < 50:  # Low confidence means better match
                name = users.get(user_id, "Unknown")
            else:
                name = "Unknown"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show the frame
        cv2.imshow("Face Recognition System", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# -------------------
# REGISTER NEW USER
# -------------------
def register_face(user_name, user_id):
    """Capture a face and register the user."""
    ret, frame = video_capture.read()
    if not ret:
        print("Error accessing the camera!")
        return

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        print("No face detected. Try again.")
        return

    # Take the first detected face
    (x, y, w, h) = faces[0]
    face = gray_frame[y:y+h, x:x+w]

    # Save face image
    face_image_path = os.path.join("face_encodings", f"{user_id}.jpg")
    cv2.imwrite(face_image_path, face)

    # Insert into DB
    insert_user(user_name, user_id, face_image_path)
    print(f"User {user_name} registered successfully!")

    # Retrain recognizer
    train_recognizer()

    # Start recognition
    recognize_and_display()

# -------------------
# MAIN EXECUTION
# -------------------
if __name__ == '__main__':
    if not os.path.exists("face_encodings"):
        os.makedirs("face_encodings")

    # Register and recognize
    user_name = input("Enter your name: ")
    user_id = input("Enter your numeric ID (e.g., 1, 2, 3): ")
    register_face(user_name, user_id)

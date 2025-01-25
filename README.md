# Face Recognition Attendance App

A simple face recognition system built with Python, OpenCV, and SQLite.  
It allows you to register new faces, save their encodings in a database, and then recognize them in real-time from your webcam.

## Features
- **Face Detection** using Haar Cascades (OpenCV).
- **Face Recognition** using LBPH (Local Binary Patterns Histograms).
- **SQLite Database** to store user names, IDs, and file paths for face images.
- **On-the-Fly Training**: Whenever a new user is registered, the recognizer is re-trained.

## Screenshot

Here’s a sample screenshot of the application detecting a face in real-time.  


![Face Recognition in Action](./Screenshot%20(173).png)


> `![Face Recognition in Action](./screenshots/Screenshot%20(173).png)`

Create and Activate a Virtual Environment (Optional but Recommended)
-->  # On Windows
python -m venv venv
venv\Scripts\activate

 --> # On macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

     The application will:
--> Ask for your name.
--> Ask for your numeric ID.
--> Capture your face from the webcam and store it.
--> Train the face recognizer.
--> Begin live recognition immediately.
--> Press q in the webcam window to quit.

attendance_app/
  ├─ face_encodings/
  │   ├─ 1.jpg
  │   ├─ 2.jpg
  │   ...
  ├─ face_database.db
  ├─ requirements.txt
  ├─ app.py
  ├─ Screenshot (173).png
  └─ README.md

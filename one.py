import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime

# Path to folder containing known 
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

# Load known faces
known_face_encodings = []
known_face_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{file}")
    encodings = face_recognition.face_encodings(image)
    if encodings:  # Ensure encoding exists
        known_face_encodings.append(encodings[0])
        known_face_names.append(os.path.splitext(file)[0])

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Function to load all past attendance records
def load_attendance():
    attendance_records = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as file:
            reader = csv.DictReader(file)
            attendance_records = list(reader)
    return attendance_records

# Function to mark login/logout
def mark_attendance(name):
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M:%S')

    attendance_records = load_attendance()

    # Check if the user has already logged in today
    found = False
    for record in attendance_records:
        if record["Name"] == name and record["Date"] == current_date:
            record["Logout Time"] = current_time  # Update logout time
            found = True
            break

    if not found:
        attendance_records.append({
            "Name": name,
            "Date": current_date,
            "Login Time": current_time,
            
        })

    # Write updated data back to the CSV
    with open(ATTENDANCE_FILE, 'w', newline='') as file:
        fieldnames = ["Name", "Date", "Login Time", "Logout Time"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(attendance_records)

# Process webcam frames
while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)  # Mark Login/Logout
        
        # Draw a rectangle and display name
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Face Recognition Attendance', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

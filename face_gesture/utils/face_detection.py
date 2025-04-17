import cv2
import face_recognition
import numpy as np
import os
from pathlib import Path

def capture_face(user_id):
    # Create faces directory if it doesn't exist
    faces_dir = Path('faces')
    user_dir = faces_dir / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return False

    captured_encodings = []
    capture_count = 0
    total_captures = 5

    print("Starting face capture process...")
    print(f"We'll take {total_captures} photos from different angles.")

    while capture_count < total_captures:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        if face_locations:
            # Draw rectangle around the face
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add instruction text
            instructions = [
                "Look straight at the camera",
                "Turn your head slightly left",
                "Turn your head slightly right",
                "Tilt your head slightly up",
                "Tilt your head slightly down"
            ]
            cv2.putText(frame, f"Capture {capture_count + 1}/5: {instructions[capture_count]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Face Registration', frame)
            
            # Wait for spacebar press to capture
            key = cv2.waitKey(1)
            if key == 32:  # Spacebar
                # Get face encoding
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings:
                    # Save the captured frame
                    capture_path = user_dir / f"{user_id}_capture_{capture_count + 1}.jpg"
                    cv2.imwrite(str(capture_path), frame)
                    
                    # Store the encoding
                    captured_encodings.append(face_encodings[0])
                    capture_count += 1
                    print(f"Capture {capture_count} successful!")
                    
                    if capture_count < total_captures:
                        print(f"\nNext: {instructions[capture_count]}")
                        print("Press spacebar when ready...")
        else:
            cv2.putText(frame, "No face detected! Please position your face in the frame.", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Face Registration', frame)
            cv2.waitKey(1)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

    if capture_count == total_captures:
        try:
            # Calculate average encoding
            average_encoding = np.mean(captured_encodings, axis=0)
            
            # Save the encoding
            encoding_file = user_dir / f"{user_id}_encoding.npy"
            np.save(str(encoding_file), average_encoding)
            print(f"\nFace registration completed! Encoding saved to {encoding_file}")
            return True
        except Exception as e:
            print(f"Error saving encoding: {str(e)}")
            return False
    else:
        print("\nFace registration incomplete!")
        return False

if __name__ == "__main__":
    user_id = input("Enter user ID: ")
    capture_face(user_id)

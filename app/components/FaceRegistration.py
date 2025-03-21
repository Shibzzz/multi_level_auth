import cv2
import face_recognition
import numpy as np
import mediapipe as mp
import time
from datetime import datetime
import os
import json

class FaceGestureRegistration:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.gesture_sequence = []
        self.required_gesture_count = 3
        self.last_gesture_time = time.time()
        self.gesture_timeout = 2.0  # seconds

    def detect_gesture(self, hand_landmarks):
        # Get thumb tip and index finger tip positions
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        # Simple gesture detection
        if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
            return "thumbs_up"
        elif thumb_tip.y > index_tip.y and thumb_tip.y > middle_tip.y:
            return "thumbs_down"
        elif abs(thumb_tip.x - index_tip.x) < 0.05:
            return "victory"
        return None

    def register_user(self, user_id):
        video_capture = cv2.VideoCapture(0)
        face_encoding = None
        registration_complete = False
        
        while not registration_complete:
            ret, frame = video_capture.read()
            if not ret:
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face Detection
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Hand Detection
            hand_results = self.hands.process(rgb_frame)
            
            # Draw face rectangles
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Process hand landmarks and gestures
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    current_gesture = self.detect_gesture(hand_landmarks)
                    current_time = time.time()
                    
                    if current_gesture and (current_time - self.last_gesture_time) > self.gesture_timeout:
                        self.gesture_sequence.append(current_gesture)
                        self.last_gesture_time = current_time
                        print(f"Gesture detected: {current_gesture}")
                        print(f"Gesture sequence: {self.gesture_sequence}")

            # Display instructions
            cv2.putText(frame, f"Gestures recorded: {len(self.gesture_sequence)}/{self.required_gesture_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Make 3 different gestures (thumbs up, thumbs down, or victory)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Registration', frame)

            # Check if we have both face and required gestures
            if len(face_locations) > 0 and not face_encoding:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings:
                    face_encoding = face_encodings[0]
                    print("Face captured successfully!")

            if face_encoding is not None and len(self.gesture_sequence) >= self.required_gesture_count:
                registration_complete = True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

        if registration_complete:
            # Save the face encoding and gesture sequence
            self.save_user_data(user_id, face_encoding, self.gesture_sequence)
            return True
        return False

    def save_user_data(self, user_id, face_encoding, gesture_sequence):
        # Create directory if it doesn't exist
        data_dir = "user_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Save face encoding
        face_data = face_encoding.tolist()  # Convert numpy array to list
        user_data = {
            "face_encoding": face_data,
            "gesture_sequence": gesture_sequence,
            "timestamp": datetime.now().isoformat()
        }

        # Save to JSON file
        file_path = os.path.join(data_dir, f"user_{user_id}.json")
        with open(file_path, 'w') as f:
            json.dump(user_data, f)
        
        print(f"User data saved successfully to {file_path}") 
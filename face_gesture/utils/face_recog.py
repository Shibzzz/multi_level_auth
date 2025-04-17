import cv2
import face_recognition
import numpy as np
import os
import logging
from pathlib import Path
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceVerifier:
    def __init__(self, model="hog"):
        """Initialize face verifier with specified model type."""
        self.model = model
        self.frame_skip = 3
        self.min_face_confidence = 0.6
        self.encoding_size = 128
        self.face_locations = []
        self.small_frame = None
        self.frame_scale = 0.25
        
        # Set up base directory
        self.faces_dir = Path('faces')
        self.faces_dir.mkdir(exist_ok=True)

    def get_user_dir(self, user_id):
        """Get the directory for a specific user's face data."""
        user_dir = self.faces_dir / str(user_id)
        if not user_dir.exists():
            logger.error(f"User directory not found: {user_dir}")
            return None
        return user_dir

    def _process_frame_for_detection(self, frame):
        """Preprocess frame for face detection."""
        self.small_frame = cv2.resize(frame, (0, 0), fx=self.frame_scale, fy=self.frame_scale)
        rgb_small_frame = cv2.cvtColor(self.small_frame, cv2.COLOR_BGR2RGB)
        return rgb_small_frame

    def _scale_face_locations(self, face_locations):
        """Scale face locations back to original size."""
        if not face_locations:
            return []
        scale = 1.0 / self.frame_scale
        return [(int(top * scale), int(right * scale), 
                int(bottom * scale), int(left * scale)) 
                for top, right, bottom, left in face_locations]

    def load_known_face(self, user_id):
        """Load known face encoding for the specified user."""
        try:
            user_dir = self.get_user_dir(str(user_id))
            if user_dir is None:
                return False
                
            encoding_path = user_dir / f"{user_id}_encoding.npy"
            if not encoding_path.exists():
                logger.error(f"No face encoding found for user {user_id}")
                return False
                
            self.known_encoding = np.load(str(encoding_path))
            logger.info(f"Successfully loaded face encoding for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading face encoding: {str(e)}")
            return False

    def process_verification_frame(self, frame):
        """Process a single frame for face verification."""
        try:
            # Process frame at reduced size
            rgb_small_frame = self._process_frame_for_detection(frame)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_small_frame, model=self.model)
            
            result = None
            distance = None
            face_location = None
            
            if face_locations:
                # Scale locations back to original size
                face_locations = self._scale_face_locations(face_locations)
                
                # Convert frame to RGB for face recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                if face_encodings:
                    # Compare with known encoding
                    results = face_recognition.compare_faces([self.known_encoding], 
                                                          face_encodings[0],
                                                          tolerance=self.min_face_confidence)
                    distance = face_recognition.face_distance([self.known_encoding], 
                                                           face_encodings[0])[0]
                    
                    result = results[0]
                    face_location = face_locations[0]
                    
                    # Draw results on frame
                    top, right, bottom, left = face_location
                    if result:
                        color = (0, 255, 0)  # Green for match
                        text = f"Match! ({distance:.2f})"
                    else:
                        color = (0, 0, 255)  # Red for no match
                        text = f"No match ({distance:.2f})"
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, text, (left, top - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            return frame, result, distance, face_location
            
        except Exception as e:
            logger.error(f"Error processing verification frame: {str(e)}")
            return frame, None, None, None

    def verify_face(self, user_id):
        """Main method to verify a face against stored encoding."""
        if not self.load_known_face(user_id):
            logger.error("Failed to load face encoding")
            return False

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open camera")
            return False

        logger.info("Starting face verification...")
        matches = []
        required_matches = 3  # Number of positive matches required
        max_attempts = 30  # Maximum frames to process

        try:
            frame_count = 0
            while frame_count < max_attempts:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame, result, distance, _ = self.process_verification_frame(frame)
                
                if result is not None:
                    matches.append(result)
                    if result:
                        logger.info(f"Match found! Distance: {distance:.2f}")
                    else:
                        logger.info(f"No match. Distance: {distance:.2f}")

                cv2.imshow('Face Verification', frame)
                frame_count += 1

                # Break if we have enough positive matches
                if matches.count(True) >= required_matches:
                    logger.info("Verification successful!")
                    break

                # Break if user presses 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        # Return True if we have enough positive matches
        success = matches.count(True) >= required_matches
        if success:
            logger.info("Face verification completed successfully")
        else:
            logger.info("Face verification failed")
        return success

if __name__ == "__main__":
    user_id = input("Enter user ID to verify: ")
    verifier = FaceVerifier()
    verifier.verify_face(user_id)

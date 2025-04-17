import cv2
import mediapipe as mp
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class GestureVerifier:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.threshold = 0.85

    def normalize_landmarks(self, landmarks):
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        center = np.mean(points, axis=0)
        scale = np.max(np.linalg.norm(points - center, axis=1))
        
        normalized = []
        for lm in landmarks:
            normalized.extend([
                (lm.x - center[0]) / scale,
                (lm.y - center[1]) / scale,
                (lm.z - center[2]) / scale
            ])
        return np.array(normalized)

    def calculate_similarity(self, gesture1, gesture2):
        similarity = np.dot(gesture1, gesture2) / (
            np.linalg.norm(gesture1) * np.linalg.norm(gesture2)
        )
        return similarity

    def load_known_gesture(self, user_id):
        """Load the saved gesture for a user."""
        # First try the new path format
        gesture_path = os.path.join("gestures", user_id, "gesture.npy")
        try:
            known_gesture = np.load(gesture_path)
            logger.info(f"Loaded gesture from {gesture_path}")
            return known_gesture
        except FileNotFoundError:
            # Fall back to the old path format
            gesture_path = os.path.join("gestures", user_id, f"{user_id}_gesture.npy")
            try:
                known_gesture = np.load(gesture_path)
                logger.info(f"Loaded gesture from {gesture_path}")
                return known_gesture
            except FileNotFoundError:
                logger.error(f"Could not find gesture for user {user_id}")
                return None

    def process_verification_frame(self, frame, known_gesture):
        """Process a single frame for gesture verification."""
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        ) as hands:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                return None, None
            
            # Normalize current landmarks
            landmarks = results.multi_hand_landmarks[0].landmark
            current_vector = self.normalize_landmarks(landmarks)
            
            # Calculate similarity with known gesture
            similarity = self.calculate_similarity(known_gesture, current_vector)
            
            return similarity, current_vector

    def verify_gesture(self, user_id):
        """Verify a gesture using webcam input."""
        known_gesture = self.load_known_gesture(user_id)
        if known_gesture is None:
            print(f"Error: Could not find gesture for user {user_id}")
            return False, "No registered gesture found"

        print("Show your gesture to verify...")
        
        cap = cv2.VideoCapture(0)
        try:
            with self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            ) as hands:
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # Flip the frame horizontally for a selfie-view display
                    frame = cv2.flip(frame, 1)
                    
                    # Convert the BGR image to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Process the image and detect hands
                    results = hands.process(image)
                    # Convert back to BGR for OpenCV
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # If hand landmarks are detected
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Draw hand landmarks
                            self.mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                            )

                            # Get normalized gesture vector
                            live_gesture = self.normalize_landmarks(hand_landmarks.landmark)
                            
                            # Calculate similarity with known gesture
                            similarity = self.calculate_similarity(known_gesture, live_gesture)
                            
                            if similarity > self.threshold:
                                match_text = f"MATCHED! Similarity: {similarity:.2%}"
                                color = (0, 255, 0)
                                
                                # Display result
                                cv2.putText(image, match_text, (10, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                cv2.imshow("Gesture Verification", image)
                                cv2.waitKey(1000)  # Show the match briefly
                                
                                return True, similarity
                            else:
                                match_text = f"NOT MATCHED. Similarity: {similarity:.2%}"
                                color = (0, 0, 255)
                                
                                # Display result
                                cv2.putText(image, match_text, (10, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        # Show help text when no hand is detected
                        cv2.putText(image, "Show your gesture in frame", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.imshow("Gesture Verification", image)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nGesture verification ended.")
        
        return False, "Verification cancelled"

# For backward compatibility
def verify_gesture_legacy(username):
    verifier = GestureVerifier()
    return verifier.verify_gesture(username)

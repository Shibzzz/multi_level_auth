import cv2
import mediapipe as mp
import numpy as np
import os
import time
import logging

logger = logging.getLogger(__name__)

class GestureRegistrar:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
    def normalize_landmarks(self, landmarks):
        # Get the hand center and scale
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        center = np.mean(points, axis=0)
        scale = np.max(np.linalg.norm(points - center, axis=1))
        
        # Normalize to be scale and translation invariant
        normalized = []
        for lm in landmarks:
            normalized.extend([
                (lm.x - center[0]) / scale,
                (lm.y - center[1]) / scale,
                (lm.z - center[2]) / scale
            ])
        return np.array(normalized)

    def calculate_consistency(self, samples):
        n_samples = len(samples)
        similarities = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                similarity = np.dot(samples[i], samples[j]) / (
                    np.linalg.norm(samples[i]) * np.linalg.norm(samples[j])
                )
                similarities.append(similarity)
        
        return np.mean(similarities), np.std(similarities)

    def process_frame(self, frame):
        """Process a single frame and return hand landmarks if detected."""
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        ) as hands:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                return None
                
            # Normalize landmarks
            landmarks = results.multi_hand_landmarks[0].landmark
            normalized_vector = self.normalize_landmarks(landmarks)
            
            return normalized_vector

    def register_gesture(self, user_id):
        """Register a gesture for a user using webcam input."""
        save_path = os.path.join("gestures", user_id)
        os.makedirs(save_path, exist_ok=True)

        # Start webcam
        cap = cv2.VideoCapture(0)
        print("Starting camera...")

        # Warm up the camera
        for _ in range(10):
            cap.read()

        print("Please show your gesture...")
        print("Hold your hand steady in frame")
        print("We'll capture 5 samples of your gesture")
        print("Press SPACE when ready to start capturing")

        samples = []
        required_samples = 5
        countdown = 3
        last_capture_time = 0
        capture_delay = 1  # 1 second between captures
        is_countdown = False
        start_time = 0

        try:
            with self.mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            ) as hands:
                
                while cap.isOpened() and len(samples) < required_samples:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # Flip the frame horizontally for a later selfie-view display
                    frame = cv2.flip(frame, 1)
                    
                    # Convert the BGR image to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Process the image and detect hands
                    results = hands.process(image)
                    # Convert back to BGR for OpenCV
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Draw hand landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                image, 
                                hand_landmarks, 
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                            )

                    # Display status
                    status_text = f"Samples captured: {len(samples)}/{required_samples}"
                    cv2.putText(image, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    current_time = time.time()

                    # Handle countdown and capture
                    if is_countdown:
                        time_elapsed = current_time - start_time
                        remaining = countdown - int(time_elapsed)
                        
                        if remaining > 0:
                            cv2.putText(image, f"Capturing in {remaining}...", (10, 70),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                        elif results.multi_hand_landmarks and current_time - last_capture_time >= capture_delay:
                            # Capture sample
                            norm_vector = self.normalize_landmarks(results.multi_hand_landmarks[0].landmark)
                            samples.append(norm_vector)
                            last_capture_time = current_time
                            
                            # Visual feedback
                            cv2.putText(image, "Captured!", (10, 70),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                            
                            if len(samples) < required_samples:
                                start_time = current_time + 0.5  # Reset countdown with a small delay
                            else:
                                is_countdown = False
                    else:
                        cv2.putText(image, "Press SPACE to start capturing", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.imshow("Gesture Registration", image)
                    key = cv2.waitKey(1)
                    
                    if key & 0xFF == ord('q'):
                        break
                    elif key & 0xFF == ord(' ') and not is_countdown:
                        is_countdown = True
                        start_time = current_time

        finally:
            cap.release()
            cv2.destroyAllWindows()

        if len(samples) == required_samples:
            # Calculate average gesture
            average_gesture = np.mean(samples, axis=0)
            # Calculate variance to check consistency
            variance = np.mean([np.sum((s - average_gesture)**2) for s in samples])
            
            if variance < 0.1:  # Threshold for consistent gestures
                save_file = os.path.join(save_path, "gesture.npy")
                np.save(save_file, average_gesture)
                
                logger.info(f"Gesture saved successfully for user {user_id}")
                logger.info(f"Saved to: {save_file}")
                logger.info(f"Gesture consistency: {(1 - variance) * 100:.1f}%")
                
                print(f"Gesture saved successfully!")
                print(f"Saved to: {save_file}")
                print(f"Gesture consistency: {(1 - variance) * 100:.1f}%")
                
                return True, save_file, (1 - variance)
            else:
                logger.error("Gestures were too inconsistent. Please try again.")
                print("Gestures were too inconsistent. Please try again.")
                return False, None, 0.0
        else:
            logger.error("Gesture registration incomplete")
            print("Gesture registration incomplete.")
            return False, None, 0.0

# For backward compatibility
def register_gesture(username):
    registrar = GestureRegistrar()
    return registrar.register_gesture(username)

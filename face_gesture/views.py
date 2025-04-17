from django.shortcuts import render
from django.http import JsonResponse
from .utils.face_recog import FaceVerifier
from .utils.gesture_detect import GestureRegistrar
from .utils.gesture_recog import GestureVerifier
import os
import cv2
import numpy as np
from PIL import Image
import io
import logging
import face_recognition
import mediapipe as mp


logger = logging.getLogger(__name__)

def process_frame_data(frame_data):
    """Convert frame data to OpenCV format."""
    try:
        # Convert the uploaded image to a PIL Image
        image = Image.open(io.BytesIO(frame_data))
        
        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return cv_image
    except Exception as e:
        logger.error(f"Error processing frame data: {str(e)}")
        return None

def register_face(request, user_id):
    """View for registering a new face."""
    if request.method == 'POST':
        if 'frame' not in request.FILES:
            return JsonResponse({
                'status': 'error',
                'message': 'No frame data received'
            }, status=400)

        try:
            # Process the frame data
            frame_data = request.FILES['frame'].read()
            frame = process_frame_data(frame_data)
            
            if frame is None:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Could not process frame data'
                }, status=400)
            
            # Get capture number
            capture_number = int(request.POST.get('capture_number', 1))
            
            # Initialize face verifier
            verifier = FaceVerifier()
            
            # Ensure the user directory exists
            user_dir = os.path.join('faces', user_id)
            os.makedirs(user_dir, exist_ok=True)
            
            # Save individual capture
            capture_path = os.path.join(user_dir, f"{user_id}_capture_{capture_number}.jpg")
            cv2.imwrite(capture_path, frame)
            
            # Process the frame for face encoding
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            if not face_locations:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No face detected in frame'
                }, status=400)
            
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if not face_encodings:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Could not generate face encoding'
                }, status=400)
            
            # Save individual encoding
            encoding_path = os.path.join(user_dir, f"{user_id}_encoding_{capture_number}.npy")
            np.save(encoding_path, face_encodings[0])
            
            # If this is the final capture, create average encoding
            if capture_number == 5:
                try:
                    # Load all encodings
                    encodings = []
                    for i in range(1, 6):
                        enc_path = os.path.join(user_dir, f"{user_id}_encoding_{i}.npy")
                        encodings.append(np.load(enc_path))
                    
                    # Calculate average encoding
                    average_encoding = np.mean(encodings, axis=0)
                    
                    # Save final encoding
                    final_encoding_path = os.path.join(user_dir, f"{user_id}_encoding.npy")
                    np.save(final_encoding_path, average_encoding)
                    
                    # Clean up individual encodings
                    for i in range(1, 6):
                        enc_path = os.path.join(user_dir, f"{user_id}_encoding_{i}.npy")
                        if os.path.exists(enc_path):
                            os.remove(enc_path)
                except Exception as e:
                    logger.error(f"Error creating average encoding: {str(e)}")
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Error creating final face encoding'
                    }, status=500)
            
            return JsonResponse({
                'status': 'success',
                'message': f'Capture {capture_number} successful'
            })
                
        except Exception as e:
            logger.error(f"Error during registration: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error during registration: {str(e)}'
            }, status=500)
    
    return render(request, 'face_gesture/register.html', {'user_id': user_id})

def verify_face(request, user_id):
    """View for verifying a face."""
    if request.method == 'POST':
        if 'frame' not in request.FILES:
            return JsonResponse({
                'status': 'error',
                'message': 'No frame data received'
            }, status=400)

        try:
            # Process the frame data
            frame_data = request.FILES['frame'].read()
            frame = process_frame_data(frame_data)
            
            if frame is None:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Could not process frame data'
                }, status=400)
            
            # Initialize face verifier
            verifier = FaceVerifier()
            
            # Load the known face encoding
            if not verifier.load_known_face(user_id):
                return JsonResponse({
                    'status': 'error',
                    'message': 'No registered face found for this user. Please register first.'
                }, status=400)
            
            # Process the frame for verification
            frame, result, distance, face_location = verifier.process_verification_frame(frame)
            
            # Convert face location to JSON-serializable format
            face_location_dict = None
            if face_location:
                top, right, bottom, left = face_location
                face_location_dict = {
                    'top': int(top),
                    'right': int(right),
                    'bottom': int(bottom),
                    'left': int(left)
                }
            
            # Convert numpy types to Python native types
            response_data = {
                'status': 'success',
                'message': 'Face processed',
                'matched': bool(result) if result is not None else False,
                'distance': float(distance) if distance is not None else None,
                'face_location': face_location_dict
            }
            
            if not face_location_dict:
                response_data['message'] = 'No face detected in frame'
            elif response_data['matched']:
                response_data['message'] = 'Face matched successfully'
            else:
                response_data['message'] = 'Face did not match'
            
            return JsonResponse(response_data)
                
        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error during verification: {str(e)}'
            }, status=500)
    
    return render(request, 'face_gesture/verify.html', {'user_id': user_id})

# def gesture_registration_page(request, user_id):
#     """View for rendering the gesture registration page."""
#     return render(request, 'face_gesture/gesture_register.html', {'user_id': user_id})

# def gesture_verification_page(request, user_id):
#     """View for rendering the gesture verification page."""
#     return render(request, 'face_gesture/gesture_verify.html', {'user_id': user_id})

def register_gesture(request):
    """API endpoint for registering a gesture."""
    if not request.method == 'POST':
        return JsonResponse({
            'status': 'error',
            'message': 'Only POST method is allowed'
        }, status=405)

    user_id = request.POST.get('user_id')
    if not user_id:
        return JsonResponse({
            'status': 'error',
            'message': 'user_id is required'
        }, status=400)

    if 'frame' not in request.FILES:
        return JsonResponse({
            'status': 'error',
            'message': 'No frame data received'
        }, status=400)

    try:
        # Process the frame data
        frame_data = request.FILES['frame'].read()
        frame = process_frame_data(frame_data)
        
        if frame is None:
            return JsonResponse({
                'status': 'error',
                'message': 'Could not process frame data'
            }, status=400)
        
        # Get capture number
        capture_number = int(request.POST.get('capture_number', 1))
        logger.info(f"Processing capture {capture_number} for user {user_id}")
        
        # Ensure the user directory exists
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        user_dir = os.path.join(base_dir, 'gestures', user_id)
        os.makedirs(user_dir, exist_ok=True)
        logger.info(f"Using directory: {user_dir}")
        
        # Process the frame for hand landmarks
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        with mp_hands.Hands(
            min_tracking_confidence=0.7,
            max_num_hands=2,
            min_detection_confidence=0.7
        ) as hands:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Convert back to BGR for OpenCV operations
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            if not results.multi_hand_landmarks:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No hand detected in frame'
                }, status=400)
            
            # Draw hand landmarks on frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Normalize and save landmarks
            landmarks = results.multi_hand_landmarks[0].landmark
            normalized_vector = GestureRegistrar().normalize_landmarks(landmarks)

            
            # Save individual landmarks with debug info
            landmarks_path = os.path.join(user_dir, f"{user_id}_landmarks_{capture_number}.npy")
            np.save(landmarks_path, normalized_vector)
            logger.info(f"Saved landmarks to: {landmarks_path}")
            
            # Save the frame with landmarks drawn
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            
            # If this is the final capture, create average landmarks
            if capture_number == 5:
                try:
                    logger.info("Processing final capture and creating average landmarks")
                    # Load all landmarks
                    all_landmarks = []
                    for i in range(1, 6):
                        lm_path = os.path.join(user_dir, f"{user_id}_landmarks_{i}.npy")
                        if not os.path.exists(lm_path):
                            raise FileNotFoundError(f"Missing landmark file: {lm_path}")
                        all_landmarks.append(np.load(lm_path))
                        logger.info(f"Loaded landmark file {i}: {lm_path}")
                    
                    # Calculate average landmarks
                    average_landmarks = np.mean(all_landmarks, axis=0)
                    
                    # Save final landmarks
                    final_landmarks_path = os.path.join(user_dir, f"{user_id}_gesture.npy")
                    np.save(final_landmarks_path, average_landmarks)
                    logger.info(f"Saved final gesture to: {final_landmarks_path}")
                    
                    # Clean up individual landmarks
                    for i in range(1, 6):
                        lm_path = os.path.join(user_dir, f"{user_id}_landmarks_{i}.npy")
                        if os.path.exists(lm_path):
                            os.remove(lm_path)
                            logger.info(f"Cleaned up temporary file: {lm_path}")
                    
                    return JsonResponse({
                        'status': 'success',
                        'message': 'Gesture registration completed successfully ',
                        #'frame': frame_data.decode('latin1')
                    })
                    
                except Exception as e:
                    logger.error(f"Error creating average landmarks: {str(e)}")
                    return JsonResponse({
                        'status': 'error',
                        'message': f'Error creating final gesture landmarks: {str(e)}'
                    }, status=500)
            
            return JsonResponse({
                'status': 'success',
                'message': f'Capture {capture_number} successful',
                'frame': frame_data.decode('latin1')
            })
                
    except Exception as e:
        logger.error(f"Error during gesture registration: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Error during registration: {str(e)}'
        }, status=500)

def normalize_landmarks(landmarks):
    """Normalize hand landmarks to be invariant to translation and scale."""
    # Convert landmarks to numpy array
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Center the points by subtracting the mean
    centered = points - np.mean(points, axis=0)
    
    # Scale to unit size
    scale = np.sqrt(np.sum(centered**2))
    normalized = centered / scale if scale != 0 else centered
    
    # Flatten to 1D vector
    return normalized.flatten()

def verify_gesture(request):
    """API endpoint for verifying a gesture."""
    if not request.method == 'POST':
        return JsonResponse({
            'status': 'error',
            'message': 'Only POST method is allowed'
        }, status=405)

    user_id = request.POST.get('user_id')
    if not user_id:
        return JsonResponse({
            'status': 'error',
            'message': 'user_id is required'
        }, status=400)

    if 'frame' not in request.FILES:
        return JsonResponse({
            'status': 'error',
            'message': 'No frame data received'
        }, status=400)

    try:
        # Process the frame data
        frame_data = request.FILES['frame'].read()
        frame = process_frame_data(frame_data)
        
        if frame is None:
            return JsonResponse({
                'status': 'error',
                'message': 'Could not process frame data'
            }, status=400)
        
        # Load the known gesture landmarks
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gesture_path = os.path.join(base_dir, 'gestures', user_id, f"{user_id}_gesture.npy")
        logger.info(f"Looking for gesture file at: {gesture_path}")
        
        try:
            known_gesture = np.load(gesture_path)
            logger.info("Successfully loaded gesture file")
        except FileNotFoundError:
            logger.error(f"No gesture file found at: {gesture_path}")
            return JsonResponse({
                'status': 'error',
                'message': 'No registered gesture found for this user. Please register first.'
            }, status=400)
        
        # Process the frame for hand landmarks
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        ) as hands:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Convert back to BGR for OpenCV operations
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            if not results.multi_hand_landmarks:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No hand detected in frame'
                }, status=400)
            
            # Draw hand landmarks on frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Normalize current landmarks
            landmarks = results.multi_hand_landmarks[0].landmark
            current_vector = normalize_landmarks(landmarks)
            
            # Calculate similarity
            similarity = np.dot(current_vector, known_gesture) / (
                np.linalg.norm(current_vector) * np.linalg.norm(known_gesture)
            )
            logger.info(f"Calculated similarity: {similarity:.2%}")
            
            # Save the frame with landmarks drawn
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            
            # Convert numpy types to Python native types
            response_data = {
                'status': 'success',
                'message': 'Gesture processed',
                'matched': bool(similarity > 0.85),
                'similarity': float(similarity),
                'frame': frame_data.decode('latin1')
            }
            
            if response_data['matched']:
                logger.info("Gesture matched successfully")
                response_data['message'] = 'Gesture matched successfully'
            else:
                logger.info("Gesture did not match")
                response_data['message'] = 'Gesture did not match'
            
            return JsonResponse(response_data)
                
    except Exception as e:
        logger.error(f"Error during gesture verification: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Error during verification: {str(e)}'
        }, status=500)

def register_gesture_direct(request, user_id):
    """View for direct gesture registration using MediaPipe."""
    if request.method == 'GET':
        return render(request, 'face_gesture/gesture_register_direct.html', {'user_id': user_id})
    else:
        # For AJAX or direct API calls
        try:
            registrar = GestureRegistrar()
            success, save_file, consistency = registrar.register_gesture(user_id)
            
            if success:
                return JsonResponse({
                    'status': 'success',
                    'message': f'Gesture registered successfully! Consistency: {consistency:.2%}',
                    'saved_to': save_file
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Gesture registration failed. Please try again.'
                }, status=400)
                
        except Exception as e:
            logger.error(f"Error during direct gesture registration: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error during registration: {str(e)}'
            }, status=500)

def verify_gesture_direct(request, user_id):
    """View for direct gesture verification using MediaPipe."""
    if request.method == 'GET':
        return render(request, 'face_gesture/gesture_verify_direct.html', {'user_id': user_id})
    else:
        # For AJAX or direct API calls
        try:
            verifier = GestureVerifier()
            success, similarity = verifier.verify_gesture(user_id)
            
            if success:
                return JsonResponse({
                    'status': 'success',
                    'message': f'Gesture verified successfully! Similarity: {similarity:.2%}'
                })
            else:
                # If similarity is a string, it's an error message
                if isinstance(similarity, str):
                    message = similarity
                else:
                    message = f'Gesture verification failed. Similarity: {similarity:.2%}'
                
                return JsonResponse({
                    'status': 'error',
                    'message': message
                }, status=400)
                
        except Exception as e:
            logger.error(f"Error during direct gesture verification: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error during verification: {str(e)}'
            }, status=500)